## Training workbook for Frontier

### Requirements

#### PyTorch-ROCm:

The PyTorch-ROCm installation instructions on the [PyTorch homepage](https://pytorch.org/get-started/locally/) work without any issues. Install PyTorch nightly with ROCm 6.2:

```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2
```


#### FlashAttention-2:
Unlike for CUDA, you will need to install FlashAttention-2 for ROCm separately. [This page](https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/model-acceleration-libraries.html) provides the instructions for that. Basically, to intall from source:

```bash
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention/
GPU_ARCHS=gfx90a python setup.py install  # MI200 series
```
Here, `gfx90a` is the correct GPU architecture choice for MI250X. In the last step, make sure to build with `ninja` (`pip install ninja`), otherwise it might take forever. Also, make sure to set your ROCm home directory correctly for the installation to proceed: *e.g.* `export ROCM_HOME=/opt/rocm-6.2.0` (or `export ROCM_HOME=/opt/rocm-6.1.3` if you're using ROCm 6.1).


The same page linked above also provides Triton kernels for FlashAttention-2, but I haven't tried them yet.

#### Hugging Face ecosystem:
You can then install the standard Hugging Face libraries in the usual way, *e.g.*:
```bash
pip install transformers datasets accelerate
```

#### `aws-ofi-rccl` plugin:
I've observed a ~20% improvement in runtime when I use the `aws-ofi-rccl` plugin, which enables `rccl` to use `libfabric`. I provide a shell script here ([`aws_ofi_rccl.sh`](https://github.com/eminorhan/frontier-guide/blob/master/aws_ofi_rccl.sh)) to install this plugin. Simply run this script (*e.g.* `sh aws_ofi_rccl.sh`) to install the plugin (the script assumes that your ROCm version is 6.2.0; if you're using a different version, change it accordingly). I should note, however, that I found this plugin to be quite unstable: it often causes jobs to fail randomly for mysterious reasons.

### Results
**Update (Sep 6):** Another ~22% improvement in runtime after I upgraded to PyTorch nightly + ROCm 6.2. I'm not sure what exactly was responsible for the improvement, but good job AMD! After upgrading PyTorch + ROCm, you will have to recompile `aws-ofi-rccl` as well with the new ROCm version (as described above). I updated the script to use ROCm 6.2. Time per update is now ~61 seconds, and estimated total time to train for 1T tokens on 64 Frontier nodes is now down to **~42 days**!

**Update (Sep 5):** A further ~20% improvement in runtime with the `aws-ofi-rccl` plugin (see above). Time per update is now ~77 seconds, and total time estimated to train for 1T tokens on 64 Frontier nodes is now down to **~53 days**. Marchons!

**Update (Sep 5):** Slightly improved throughput by freeing some memory before the backward pass (inspired by the [torchtitan](https://github.com/pytorch/torchtitan) repo). Batch size is now 4 per device, tokens per update is 16.8M, time per update is ~95 seconds, and total time estimated to train for 1T tokens on 64 Frontier nodes is now **~65 days**.

The training files in this repo ([`train.py`](https://github.com/eminorhan/frontier-guide/blob/master/train.py) and [`train.sh`](https://github.com/eminorhan/frontier-guide/blob/master/train.sh)) are my first attempt to train a Llama-3.1 8B model with a context length of 8192 on 64 Frontier nodes (256 MI250Xs with a total of 512 "compute dies" or "GPUs"). The batch size is 4 per device (2048 globally), so each update consumes 2048 * 8192 = 16.8M tokens in total. Currently, each training update takes around 95 seconds to complete. I calculated that it would take around 65 days to train this model over 1T tokens in this particular configuration (*i.e.* 64 nodes). The implementation currently uses FlashAttention-2, FSDP (full-shard), automatic mixed precision training (with `bf16`), and activation checkpointing. I'm not quite sure yet how suboptimal these results are.

### TODO:

- [ ] Write a version of the training code without `accelerate` for greater control.
- [ ] Try the [torchtitan](https://github.com/pytorch/torchtitan) codebase. 
- [ ] Try MosaicML's [llm-foundry](https://github.com/mosaicml/llm-foundry) codebase.
- [ ] Try the [TinyLlama](https://github.com/jzhang38/TinyLlama) codebase.
