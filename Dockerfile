# Production Multi-Stage / Cloud-Ready Dockerfile for NeuroChess v3.0 (PyTorch CPU + Stockfish NNUE + Lc0)
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    TORCH_NUM_THREADS=2 \
    NEUROCHESS_WEIGHTS_DIR=/app/backend/weights

# Create non-root user for Kubernetes PodSecurity compliance
RUN groupadd -r neurochess && useradd -r -g neurochess -u 1000 neurochess

COPY backend/requirements.txt /app/backend/requirements.txt

# Install lightweight CPU-only PyTorch first so CUDA libraries are never pulled into cloud containers
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.2.0" && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend (including pre-trained PyTorch weights) & frontend
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

RUN mkdir -p /app/backend/weights && chown -R neurochess:neurochess /app
USER 1000:1000

WORKDIR /app/backend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
