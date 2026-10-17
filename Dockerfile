# Timestral × M5 demo app, ready for Google Cloud Run (or any container host).
# The M5 data and the Timestral weights are downloaded at build time, so the container starts quickly.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    GRADIO_ANALYTICS_ENABLED=False \
    GRADIO_SERVER_NAME=0.0.0.0

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-app.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-app.txt

COPY app.py app_texts.py download_data.py run_baselines.py ./
COPY m5bench ./m5bench
COPY app_assets ./app_assets

# M5 data (public copy by Nixtla) and the winning submission used for comparison
RUN python download_data.py --data data --top 1
# Timestral weights: the two fast sizes are baked in; 123M and 209M download on first use
RUN python -c "from huggingface_hub import snapshot_download as d; \
[d(r, allow_patterns=['config.json', 'model.safetensors']) for r in ('mistralai/Timestral-28M', 'mistralai/Timestral-5M')]" \
    || echo "Weights not baked in; they will download on first use"

# Cloud Run sets PORT (8080 by default)
CMD ["python", "app.py"]
