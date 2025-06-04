FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# Fix the copy paths - this is the key change
COPY app/ ./app/
COPY display/ ./display/

# Debug to verify
RUN echo "=== Files in /app ===" && ls -la
RUN echo "=== Frontend check ===" && ls -la frontend/ || echo "No frontend dir"

RUN mkdir -p uploads data models

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]