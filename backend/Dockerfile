# Use the official FastAPI image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install dependencies
RUN apt-get update && \
    apt-get install -y poppler-utils tesseract-ocr libmagic-dev libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/* && \
    pip install -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY ./src /app/src

# Expose port 80 to the outside world
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]