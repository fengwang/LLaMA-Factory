# Docker Setup for NVIDIA GPUs

This directory contains Docker configuration files for running LLaMA Factory with NVIDIA GPU support.

For **GeForce RTX 50-series GPUs**, including the RTX 5090, follow [the consumer Blackwell instructions](#geforce-rtx-50-series-consumer-blackwell). The default Dockerfile and Compose configuration use CUDA 12.4 and do not support these GPUs.

## Prerequisites

### Linux-specific Requirements

Before running the Docker container with GPU support, you need to install the following packages:

1. **Docker**: The container runtime
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install docker.io

   # Or install Docker Engine from the official repository:
   # https://docs.docker.com/engine/install/
   ```

2. **Docker Compose** (if using the docker-compose method):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install docker-compose

   # Or install the latest version:
   # https://docs.docker.com/compose/install/
   ```

3. **NVIDIA Container Toolkit** (required for GPU support):
   ```bash
   # Add the NVIDIA GPG key and repository
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   # Install nvidia-container-toolkit
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit

   # Restart Docker to apply changes
   sudo systemctl restart docker
   ```

   **Note**: Without `nvidia-container-toolkit`, the Docker container will not be able to access your NVIDIA GPU.

### Verify GPU Access

After installation, verify that Docker can access your GPU:

```bash
sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If successful, you should see your GPU information displayed.

## Usage

### Using Docker Compose (Recommended)

```bash
cd docker/docker-cuda/
docker compose up -d
docker compose exec llamafactory bash
```

### Using Docker Run

```bash
# Build the image
docker build -f ./docker/docker-cuda/Dockerfile \
    --build-arg PIP_INDEX=https://pypi.org/simple \
    --build-arg EXTRAS=metrics \
    -t llamafactory:latest .

# Run the container
docker run -dit --ipc=host --gpus=all \
    -p 7860:7860 \
    -p 8000:8000 \
    --name llamafactory \
    llamafactory:latest

# Enter the container
docker exec -it llamafactory bash
```

## GeForce RTX 50-series (consumer Blackwell)

Use [Dockerfile.rtx-50-series](Dockerfile.rtx-50-series) for Linux x86_64 systems with consumer Blackwell GPUs (compute capability `12.0`, or `sm_120`). This image includes:

| Component | Version |
| --- | --- |
| Base image | `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` |
| Python | 3.11 |
| CUDA | 12.8 |
| PyTorch / torchvision / torchaudio | 2.8.0 / 0.23.0 / 2.8.0 |
| FlashAttention 2 | 2.8.3, compiled for `sm_120` |

[FA2 2.8.3 supports compiling for `sm_120` with CUDA 12.8 or newer](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/setup.py). The Dockerfile builds it from source against the image's PyTorch ABI and pins the matching PyTorch packages during LLaMA-Factory installation. It does not install FlashAttention 3. This image targets RTX 50-series GPUs; use the default image for older GPU architectures.

### Prerequisites and build

Install a host NVIDIA driver that supports your RTX 50-series GPU and [CUDA 12.8](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html), plus the NVIDIA Container Toolkit described above. The CUDA toolkit comes from the image; a host CUDA installation is not required. Check container GPU access with:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

Build from the repository root:

```bash
docker build -f docker/docker-cuda/Dockerfile.rtx-50-series \
    --build-arg PIP_INDEX=https://pypi.org/simple \
    --build-arg MAX_JOBS=4 \
    -t llamafactory:rtx-50-series .
```

The first build downloads a large development image and compiles FA2, so allow time and sufficient disk space and RAM. `MAX_JOBS` controls FA2 compilation parallelism; reduce it to `2` or `1` if the build runs out of memory. Compilation targets `sm_120` explicitly and does not require a GPU during `docker build`. GPU access is required for the tests and training below.

### Run

```bash
docker run --rm -it --ipc=host --gpus all \
    -p 7860:7860 -p 8000:8000 \
    llamafactory:rtx-50-series bash
```

Inside the container, use `llamafactory-cli webui` for LLaMA Board or `llamafactory-cli train` for training. Set `flash_attn: fa2` and `bf16: true` (or `fp16: true`) in your training configuration to select FA2 explicitly.

For Compose, change `services.llamafactory.build.dockerfile` in your local `docker-compose.yml` to `./docker/docker-cuda/Dockerfile.rtx-50-series`, then run `docker compose up -d --build` from `docker/docker-cuda/`. The repository's default Compose configuration continues to select the CUDA 12.4 image.

### Minimal GPU smoke test

Run this after building. It calls FA2 directly, checks FP16 and BF16 forward and backward results against PyTorch's FP32 math attention, and covers grouped-query attention with head dimensions 64 and 256. An import or `nvidia-smi` check alone does not establish that the CUDA kernels work.

```bash
docker run --rm -i --gpus all --ipc=host llamafactory:rtx-50-series python - <<'PY'
import torch
import torch.nn.functional as F
import flash_attn
from flash_attn import flash_attn_func
from torch.nn.attention import SDPBackend, sdpa_kernel

assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.get_device_capability() == (12, 0), "Expected an RTX 50-series GPU"
assert flash_attn.__version__ == "2.8.3"
print(torch.cuda.get_device_name(), torch.__version__, torch.version.cuda, flash_attn.__version__)
torch.manual_seed(0)
for dtype in (torch.float16, torch.bfloat16):
    for head_dim in (64, 256):
        q, k, v = [
            torch.randn(1, 64, heads, head_dim, device="cuda", dtype=dtype, requires_grad=True)
            for heads in (4, 2, 2)
        ]
        refs = [x.detach().float().requires_grad_(True) for x in (q, k, v)]
        out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
        with sdpa_kernel(SDPBackend.MATH):
            ref = F.scaled_dot_product_attention(
                *(x.transpose(1, 2) for x in refs), is_causal=True, enable_gqa=True
            ).transpose(1, 2)
        grad = torch.randn_like(out)
        out.backward(grad)
        ref.backward(grad.float())
        tol = 0.03 if dtype == torch.bfloat16 else 0.005
        for actual, expected in [(out, ref)] + [(x.grad, r.grad) for x, r in zip((q, k, v), refs)]:
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual.float(), expected, atol=tol, rtol=tol)
        torch.cuda.synchronize()
        print(f"PASS: {dtype}, head_dim={head_dim}, FA2 forward/backward")
PY
```

If this reports an undefined symbol, check that PyTorch was not replaced after FA2 was built. Rebuild the image to restore the matching versions. A `no kernel image is available` error usually means an image or extension was built without support for the GPU architecture.

### Optional tiny-model training test

With a local copy of Qwen3.5-0.8B, run two LoRA training steps on the bundled demo dataset. Qwen3.5 also needs `flash-linear-attention` for its linear-attention layers in LLaMA-Factory's FA2 training path. Install that model-specific dependency in this temporary container; it is separate from FlashAttention 2.

Run from the repository root, replacing `MODEL_PATH` with your model directory:

```bash
MODEL_PATH=/path/to/Qwen3.5-0.8B
docker run --rm -i --gpus all --ipc=host \
    -v "${MODEL_PATH}:/model:ro" \
    llamafactory:rtx-50-series bash -s <<'SH'
set -eu
pip install --no-cache-dir "torch==2.8.0" "flash-attn==2.8.3" "flash-linear-attention==0.4.1"
llamafactory-cli train \
    --model_name_or_path /model \
    --stage sft --do_train true --finetuning_type lora --lora_target q_proj,v_proj \
    --dataset alpaca_en_demo --dataset_dir /app/data \
    --template qwen3_5_nothink --enable_thinking false \
    --cutoff_len 128 --max_samples 4 --max_steps 2 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
    --learning_rate 0.0001 --bf16 true --flash_attn fa2 \
    --output_dir /tmp/rtx50-smoke --logging_steps 1 --save_strategy no \
    --report_to none --preprocessing_num_workers 1 --dataloader_num_workers 0
SH
```

Check that the log selects FlashAttention 2 and completes both steps with finite loss. The model is mounted read-only; training output is discarded when the container exits. This checks training integration, not model quality or a performance speedup.

## Troubleshooting

### GPU Not Detected

If your GPU is not detected inside the container:

1. Ensure `nvidia-container-toolkit` is installed
2. Check that the Docker daemon has been restarted after installation
3. Verify your NVIDIA drivers are properly installed: `nvidia-smi`
4. Check Docker GPU support: `docker run --rm --gpus all ubuntu nvidia-smi`

### Permission Denied

If you get permission errors, ensure your user is in the docker group:

```bash
sudo usermod -aG docker $USER
# Log out and back in for changes to take effect
```

## Megatron Bridge Image

`Dockerfile.megatron` builds a CUDA runtime for LLaMA-Factory + [Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/):

| Component | Version |
| --- | --- |
| Base | `ubuntu:22.04` (Python 3.12) |
| PyTorch | 2.12.1+cu126 (CUDA libs from wheels) |
| TransformerEngine | 2.17.0 |
| megatron-core | 0.18.x (via megatron-bridge) |
| megatron-bridge | 0.5.0 |

### Build

From repo root:

```bash
docker build -f docker/docker-cuda/Dockerfile.megatron \
  -t llamafactory-megatron-bridge:latest .
```

### Run training

```bash
docker run --rm -it --gpus all --ipc=host --shm-size=16g \
  -e DISABLE_VERSION_CHECK=1 \
  -e USE_MEGATRON_BRIDGE=1 \
  -v "$PWD":/app -w /app \
  llamafactory-megatron-bridge:latest
```

## Additional Notes

- The default image is built on Ubuntu 22.04 (x86_64), CUDA 12.4, Python 3.11, PyTorch 2.6.0, and Flash-attn 2.7.4
- For different CUDA versions, you may need to adjust the base image in the Dockerfile
- Make sure your NVIDIA driver version is compatible with the CUDA version used in the Docker image
