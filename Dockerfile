# Base image (example)
FROM python:3.11-slim

# Install system deps for OCR + PDFs
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN echo "==== requirements in image ====" && cat requirements.txt && \
    pip install --no-cache-dir --no-deps -r requirements.txt

# Copy app code
COPY . .

EXPOSE 7860

# Run only the UI
CMD ["python", "ui_app.py"]