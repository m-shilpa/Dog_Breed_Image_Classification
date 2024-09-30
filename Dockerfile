# Dockerfile.train
FROM python:3.8-slim

# Set the working directory
WORKDIR /workspace

COPY requirements.txt requirements.txt

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the necessary files
COPY . .
