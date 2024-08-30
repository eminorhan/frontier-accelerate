## Training workbook for Frontier

### Requirements

#### PyTorch-ROCm:

The PyTorch-ROCm installation instructions on the [PyTorch homepage](https://pytorch.org/get-started/locally/) work without any issues:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
```

AMD provides wheels for a newer version of ROCm (6.2) [here](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html), but this requires a nightly build of PyTorch.

#### FlashAttention-2:
Unlike for CUDA, you will need to install FlashAttention-2 for ROCm separately. [This page](https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/model-acceleration-libraries.html) provides the instructions for that. Basically, to intall from source:

```bash
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention/
GPU_ARCHS=gfx90a python setup.py install  # MI200 series
```
Here, `gfx90a` is the correct GPU architecture choice for MI250X. In the last step, make sure to build with `ninja` (`pip install ninja`), otherwise it might take forever. The same page linked above also provides Triton kernels for FlashAttention-2, but I haven't tried them yet.

#### Hugging Face ecosystem:
You can then install the standard Hugging Face libraries in the usual way, *e.g.*:
```bash
pip install transformers datasets accelerate
```

### Results
The training files in this repo ([`train.py`](https://github.com/eminorhan/frontier-guide/blob/master/train.py) and [`train.sh`](https://github.com/eminorhan/frontier-guide/blob/master/train.sh)) are my first attempt to train a Llama-3.1 8B model with a context length of 8192 on 64 Frontier nodes (256 MI250Xs with a total of 512 "compute dies" or "GPUs"). The batch size is 3 per device (1536 globally), so each update consumes 1536 * 8192 = 12.6M tokens in total. Currently, each training update takes around 90 seconds to complete. I calculated that it would take around 80 days to train this model over 1T tokens in this particular configuration (*i.e.* 64 nodes). The implementation currently uses FlashAttention-2, FSDP (full-shard), automatic mixed precision training (with `bf16`), and activation checkpointing. I'm not quite sure yet how suboptimal these results are.

### TODO:

- [ ] Write a version of the training code without `accelerate` for greater control.
- [ ] Try the [torchtitan](https://github.com/pytorch/torchtitan) codebase. 
- [ ] Try MosaicML's [llm-foundry](https://github.com/mosaicml/llm-foundry) codebase.
- [ ] Try the [TinyLlama](https://github.com/jzhang38/TinyLlama) codebase.
