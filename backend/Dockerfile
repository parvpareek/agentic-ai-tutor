# Use Python 3.11 slim image
FROM python:3.11-slim

# No system dependencies needed anymore (removed sentence-transformers)
# PyMuPDF works with Python-only dependencies

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

