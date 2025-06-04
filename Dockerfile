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

RUN echo "=== Starting copy process ==="

COPY app/ ./app/
RUN echo "=== App directory copied ===" && ls -la app/

COPY frontend/ ./frontend/
RUN echo "=== Frontend directory copied ===" && ls -la frontend/

RUN echo "=== Complete app structure ===" && find . -type f | head -20

RUN echo "=== Checking frontend/index.html ===" && \
    if [ -f "./frontend/index.html" ]; then \
        echo "frontend/index.html EXISTS" && \
        echo "File size: $(wc -c < ./frontend/index.html) bytes"; \
    else \
        echo "frontend/index.html NOT FOUND"; \
    fi

RUN mkdir -p uploads data models

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]