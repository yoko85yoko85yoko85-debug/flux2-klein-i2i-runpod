FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir \
    runpod==1.7.9 \
    diffusers==0.37.1 \
    transformers==4.57.3 \
    accelerate==1.12.0 \
    huggingface-hub==0.36.0 \
    safetensors==0.5.3 \
    Pillow==11.2.1
COPY handler.py /app/handler.py
ENV HF_HOME=/runpod-volume/huggingface
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
CMD ["python", "-u", "/app/handler.py"]
