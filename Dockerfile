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

COPY app/ ./app/

COPY frontend/ ./frontend/

COPY railway.toml .
COPY .env .

RUN echo "=== App structure ===" && find . -type f -name "*.py" | head -10
RUN echo "=== Frontend files ===" && find ./frontend -type f | head -10 || echo "No frontend files found"

RUN mkdir -p uploads data models

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]