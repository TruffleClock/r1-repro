python --version && python -c "import torch; print(torch.__version__)" && nvcc --version

git clone https://github.com/TruffleClock/r1-repro

cd r1-repro
python -m venv venv
source venv/bin/activate

pip install "torch==2.5.1" tensorboard "setuptools<71.0.0"  --index-url https://download.pytorch.org/whl/cu118
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