FROM python:3.11-slim

WORKDIR /app

# system dependencies needed by pymupdf, pillow, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# directories for runtime data (mounted as volumes in production,
# but created here so the app doesn't fail if they're missing)
RUN mkdir -p data/pdfs data/processed data/chroma_db

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]