# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies needed for sentence-transformers and PyMuPDF
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Railway will set PORT env var)
EXPOSE ${PORT:-8000}

# Run the application (Railway provides PORT env var)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}

