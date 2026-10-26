# Multi-Service Production Container for ML-NIDS & XAI SOC Console
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install lightweight system dependencies (libpcap for packet processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY src/ src/
COPY artifacts/ artifacts/
COPY configs/ configs/
COPY reports/ reports/
COPY web/ web/
COPY run_app.py .
COPY README.md .

# Expose ports: 8000 (FastAPI Web Console) & 8501 (Streamlit SOC Console)
EXPOSE 8000 8501

# Default command: launch the FastAPI REST API and Web Console
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
