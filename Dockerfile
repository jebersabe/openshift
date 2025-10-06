# Use Ubuntu 22.04 LTS as base image
FROM ubuntu:22.04

# Set maintainer label
LABEL maintainer="jebersabe@gmail.com"

# Update package list and install basic utilities
RUN apt-get update && \
    apt-get install -y \
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python 3.10 and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3.10-venv && \
    rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python3 -m venv /venv

# Copy inference.py to /app directory
COPY inference.py /app/
COPY requirements.txt /app/

# Install Python dependencies
RUN /venv/bin/pip install -r requirements.txt

# Create a non-root user
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Default command
CMD ["/venv/bin/python3", "inference.py"]
