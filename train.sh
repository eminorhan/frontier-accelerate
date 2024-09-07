#!/bin/bash

#SBATCH --account=stf218
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --job-name=train_llama
#SBATCH --output=train_llama_%A_%a.out
#SBATCH --array=0
#SBATCH --qos=debug

# set proxy server to enable communication with outside
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

# enable aws-ofi-rccl
export LD_LIBRARY_PATH=/lustre/orion/stf218/scratch/emin/aws-ofi-rccl/lib:$LD_LIBRARY_PATH
export NCCL_NET_GDR_LEVEL=3   # Can improve performance, but remove this setting if you encounter a hang/crash.
export NCCL_ALGO=TREE         # May see performance difference with either setting. (should not need to use this, but can try)
export NCCL_CROSS_NIC=1       # On large systems, this NCCL setting has been found to improve performance

# honestly, not sure how many of these below (if any) are absolutely necessary
# export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_IB_TIMEOUT=31

export HF_HOME="/lustre/orion/stf218/scratch/emin/huggingface"
export HF_DATASETS_CACHE="/lustre/orion/stf218/scratch/emin/huggingface"

# root model directory
MODEL_ROOT_DIR="/lustre/orion/stf218/scratch/emin/frontier-guide"
SP="llama"

export GPUS_PER_NODE=8

# set network
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --main_process_ip $head_node_ip \
    --main_process_port 3442 \
    --machine_rank $SLURM_NODEID \
    --mixed_precision bf16 \
    --rdzv_backend c10d \
    --use_fsdp \
    --fsdp_auto_wrap_policy SIZE_BASED_WRAP \
    --fsdp_backward_prefetch BACKWARD_PRE \
    --fsdp_min_num_params 2000 \
    --fsdp_sharding_strategy 1 \
    --fsdp_state_dict_type FULL_STATE_DICT \
    "
export SCRIPT="/lustre/orion/stf218/scratch/emin/frontier-guide/train.py"
export SCRIPT_ARGS=" \
    --model_name_or_path "meta-llama/Meta-Llama-3.1-8B" \
    --block_size 8192 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 0.0001 \
    --output_dir "${MODEL_ROOT_DIR}/${SP}" \
    --num_train_epochs 20 \
    --checkpointing_steps 100 \
    "
# this step is necessary because accelerate launch does not seem to handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 
srun $CMD

echo "Done"