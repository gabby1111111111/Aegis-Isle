# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/opt/poetry-cache \
    POETRY_HOME="/opt/poetry" \
    POETRY_VERSION=1.7.1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-chi-sim \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml requirements.txt ./

# Install Python dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-dev

# Alternative: Use pip if poetry is not preferred
# RUN pip install --no-cache-dir -r requirements.txt

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create necessary directories
RUN mkdir -p /app/data/uploads /app/data/processed /app/logs /app/models/local \
    && chown -R appuser:appuser /app

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command
CMD ["uvicorn", "src.aegis_isle.api.main:app", "--host", "0.0.0.0", "--port", "8000"]