FROM python:3.9-slim

# First set WORKDIR to / for frontend copy
WORKDIR /

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy frontend from project root to /frontend in container
COPY frontend/ /frontend/
RUN echo "=== Frontend files ===" && \
    find /frontend -type f | head -20 && \
    chmod -R a+rX /frontend

# Now set WORKDIR for the app
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# Copy app code
COPY app/ ./app/

# Create writable directories
RUN mkdir -p uploads data models && \
    chmod -R a+rwX uploads data models

# Verify the final structure
RUN echo "=== Final structure ===" && \
    echo "/frontend contents:" && ls -la /frontend && \
    echo "/app contents:" && ls -la /app && \
    [ -f "/frontend/index.html" ] && \
    echo "Frontend index.html exists (size: $(wc -c < /frontend/index.html) bytes" || \
    echo "ERROR: Frontend index.html missing!"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]