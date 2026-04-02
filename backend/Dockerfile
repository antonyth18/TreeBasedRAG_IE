FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies if required (mostly for Faiss/numpy/building wheels if needed)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt .

# Install python dependencies. (FastAPI extensions recommended by README)
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir python-multipart uvicorn fastapi pydantic pydantic-settings
RUN python -m spacy download en_core_web_sm

# Copy project files
COPY . .

# Set python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Start FastAPI server on all interfaces
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
