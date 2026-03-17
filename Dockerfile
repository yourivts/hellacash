FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install Python 3.11 and system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    gcc libpq-dev && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install CUDA-enabled torch for GPU acceleration (RTX 2060)
RUN pip install --no-cache-dir --break-system-packages torch --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY . .

CMD ["python", "-m", "bot.main"]
