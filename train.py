#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""

import argparse
import logging
import math
import os
import random
from itertools import chain

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler
)
from transformers.utils.versions import require_version
from datetime import timedelta
from accelerate import InitProcessGroupKwargs

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
os.environ['CURL_CA_BUNDLE'] = ''

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune large language models on causal language modeling tasks")
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.", required=False)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="cosine", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=100, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--block_size", type=int, default=8192, help="Context length.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--checkpointing_steps", type=str, default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="If the training should continue from a checkpoint folder.")
    parser.add_argument("--use_pretrained_weights", action="store_true", help="Whether to use pretrained weights.")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    print(args)

    # context length (default: 8192)
    block_size = args.block_size  # tokenizer.model_max_length

    # Initialize the accelerator
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))  # 1 hour
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[process_group_kwargs])
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # In distributed training, 'load_dataset' function guarantees that only one local process can concurrently download the dataset.
    logger.info(f"Loading dataset.")
    with accelerator.main_process_first():
        raw_datasets = load_dataset("Salesforce/wikitext", "wikitext-2-v1", trust_remote_code=True)  # TODO: pass dataset as argument rather than hardcoding

    # Load pretrained config, model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, token=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, model_max_length=block_size, token=True)  # TODO: pass this more beautifully
    if args.model_name_or_path.startswith("meta-llama") or args.model_name_or_path.startswith("gpt2") or args.model_name_or_path.startswith("EleutherAI"):
        tokenizer.pad_token = tokenizer.eos_token

    if args.model_name_or_path and args.use_pretrained_weights:
        logger.info("Loading pretrained weights")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16, from_tf=bool(".ckpt" in args.model_name_or_path), config=config, token=True)
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)

    model.resize_token_embeddings(new_num_tokens=len(tokenizer), pad_to_multiple_of=128)
    logger.info(f"Tokenizer (vocab) size: {len(tokenizer)}")
    logger.info(f"Pad token id: {tokenizer.pad_token_id}")

    # Turn on gradient checkpointing
    model.gradient_checkpointing_enable()

    # Preprocessing the datasets. First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function_original(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function_original,
            batched=True,
            num_proc=None,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    def preprocess_function_original(examples):
        # Concatenate all texts (aka sequence packing).
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can customize this part to your needs
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()}
        result["labels"] = result["input_ids"].copy()
        return result
    
    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            preprocess_function_original,
            batched=True,
            num_proc=None,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping text",
        )

    # dataset & dataloader creation
    train_dataset = lm_datasets["train"]
    val_dataset = lm_datasets["validation"]
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size)

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        logger.info(f"Sample {index} of the training set (decoded): {tokenizer.decode(train_dataset[index]['input_ids'], skip_special_tokens=True)}.")

    logger.info(f"Model = {model}")
    logger.info(f"Number of params (M): {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1.e6}")
    model = accelerator.prepare(model)

    # Optimizer: split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Prepare lr scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    )

    # Prepare everything with our `accelerator`.
    optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, val_dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    # initialize loss counters
    train_loss = 0
    val_loss = 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # print(loss.get_device())  # check to see what device this is on
                # keep track of the train loss
                train_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.set_description(f"lr: {lr_scheduler.scheduler.get_last_lr()[0]}")
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                # the last clause makes sure checkpointing is done once only with gradient accumulation
                if completed_steps % checkpointing_steps == 0 and completed_steps != 0 and step % args.gradient_accumulation_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)

                    # save model and tokenizer
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(output_dir)

                    # log train_loss & re-initialize
                    logger.info(f"Mean train loss: {train_loss.item() / (checkpointing_steps * args.gradient_accumulation_steps)}")
                    train_loss = 0

                    # save val losses 
                    model.eval()
                    for step, batch in enumerate(val_dataloader):
                        with torch.no_grad():
                            outputs = model(**batch)
                            loss = outputs.loss
                            # keep track of the val loss
                            val_loss += loss.detach().float()

                    # log val & re-initialize
                    logger.info(f"Mean val loss: {val_loss.item() / len(val_dataloader)}")
                    val_loss = 0

            if completed_steps >= args.max_train_steps:
                break

if __name__ == "__main__":
    main()