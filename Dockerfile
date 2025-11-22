# Lightweight Python image for reproducible demos
FROM python:3.11-slim

# Install system dependencies for scientific Python
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement specification and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Default command runs the demo training loop
CMD ["python", "app/main.py"]
