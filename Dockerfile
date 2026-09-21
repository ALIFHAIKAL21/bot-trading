# Multi-Model Trading Bot & Dashboard Dockerfile
FROM python:3.10-slim

# Set working directory & environment variables
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager for fast, reliable installs
RUN pip install --no-cache-dir uv

# Copy dependencies first for caching
COPY requirements.txt pyproject.toml ./
RUN uv pip install --no-cache-dir --system -r requirements.txt

# Copy application source code, models, and config
COPY config/ config/
COPY src/ src/
COPY scripts/ scripts/
COPY models_store/ models_store/
COPY run.py .env.example ./

# Create persistent data and reports directories
RUN mkdir -p data/cache reports logs

# Expose Streamlit dashboard (8502) and REST API (8000)
EXPOSE 8502 8000

# Default entrypoint: runs scalper in background and dashboard in foreground
CMD ["sh", "-c", "python run.py --scalp-live > logs/scalper.log 2>&1 & python -m streamlit run src/dashboard/app.py --server.port 8502 --server.address 0.0.0.0 --server.headless true"]
