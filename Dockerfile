FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      tesseract-ocr \
      libgl1-mesa-glx \
      libsm6 libxext6 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and processing script
COPY models/ ./models/
COPY process_pdfs.py .

# Default command: process PDFs with 8 workers
CMD ["python","process_pdfs.py",\
     "--model","models/best.onnx",\
     "--input_dir","/app/input",\
     "--output_dir","/app/output",\
     "--max_workers","8"]