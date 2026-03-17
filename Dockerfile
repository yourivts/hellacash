FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install system deps (Python 3.10 is already in the CUDA image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev gcc libpq-dev && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install CUDA-enabled torch for GPU acceleration (RTX 2060)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "bot.main"]
