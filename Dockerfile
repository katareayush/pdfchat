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

# Copy from project root to container paths
COPY app/ ./app/                    # Copy app/ to /app/app/
COPY frontend/ ./frontend/          # Copy frontend/ to /app/frontend/

# Debug: Verify structure
RUN echo "=== Container structure ===" && \
    ls -la && \
    echo "=== App directory ===" && \
    ls -la app/ && \
    echo "=== Frontend directory ===" && \
    ls -la frontend/

# Create required directories  
RUN mkdir -p uploads data models

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]