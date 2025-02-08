#!/bin/bash
set -e

# git clone https://github.com/TruffleClock/r1-repro
# cd r1-repro

python --version && python -c "import torch; print(torch.__version__)" && nvcc --version

read -p "Enter CUDA version (e.g., 12.4 for CUDA 12.4): " cuda_version

if ! [[ "$cuda_version" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Please enter a valid CUDA version (e.g., 12.4)."
    exit 1
fi

cuda_tag="cu$(echo "$cuda_version" | tr -d '.')"
TORCH_INDEX_URL="https://download.pytorch.org/whl/$cuda_tag"

echo "Using PyTorch index: $TORCH_INDEX_URL"

python -m venv venv
source venv/bin/activate

pip install "torch==2.5.1" "setuptools<71.0.0"  --index-url "$TORCH_INDEX_URL"
pip install tensorboard

pip install flash-attn

pip install  --upgrade \
  "transformers==4.48.1" \
  "datasets==3.1.0" \
  "accelerate==1.3.0" \
  "hf-transfer==0.1.9" \
  "deepspeed==0.15.4" \
  "trl==0.14.0"

pip install "vllm==0.7.0"

pip install huggingface_hub
git config --global credential.helper store
huggingface-cli login