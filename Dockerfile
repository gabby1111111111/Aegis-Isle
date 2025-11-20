# Aegis Isle Multi-Agent RAG System
# Production-Ready Docker Container Configuration

FROM ubuntu:22.04

# Set environment variables for better performance and security
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONPATH=/app/src

# Create app user and directories for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN mkdir -p /app /app/logs /app/data /app/uploads /app/models && chown -R appuser:appuser /app

# Install system dependencies including OCR and graphics libraries
RUN apt-get update && apt-get install -y \
    # Python and build tools
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    pkg-config \
    # OCR dependencies (required for document processing)
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    poppler-utils \
    # Graphics and video libraries (required for image processing)
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libfontconfig1 \
    libfreetype6-dev \
    # Network and utilities
    curl \
    wget \
    git \
    # Clean up to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Set working directory
WORKDIR /app

# Create Python virtual environment
RUN python3.11 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip and install essential packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better Docker layer caching
COPY requirements.txt /app/

# Install Python dependencies in optimized order
# Install PyTorch first (largest dependency)
RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/src/
COPY .env.example /app/.env.example

# Create required directories and set permissions
RUN mkdir -p /app/logs /app/data/uploads /app/data/processed /app/models/local \
    && chown -R appuser:appuser /app \
    && chmod -R 755 /app

# Switch to non-root user for security
USER appuser

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose application port
EXPOSE 8000

# Run the FastAPI application with production settings
CMD ["python", "-m", "uvicorn", "src.aegis_isle.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--access-log"]