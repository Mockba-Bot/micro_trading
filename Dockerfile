# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for ta-lib
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    libssl-dev \
    libffi-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install ta-lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create necessary directories
RUN mkdir -p /app/trained_models /app/files /app/operations

# Expose the port that the FastAPI app runs on
EXPOSE 8001

# Command to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]