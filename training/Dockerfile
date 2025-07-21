# CUDA base with PyTorch wheels pre‑installed
FROM nvcr.io/nvidia/pytorch:24.05-py3

ARG HF_HOME=/opt/hf
ENV  HF_HOME=${HF_HOME} \
     TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
     PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Core libraries for LoRA + DeepSpeed
RUN pip install --no-cache-dir \
        "transformers[torch]==4.41.0" \
        datasets==2.19.0 \
        accelerate==0.30.1 \
        peft==0.11.1 \
        deepspeed==0.14.2 \
        bitsandbytes==0.43.1 \
        sentencepiece==0.2.0 \
        tqdm tensorboard

WORKDIR /training
COPY . .
ENV OUTPUT_DIR=/checkpoints
ENTRYPOINT ["python", "train.py"]

