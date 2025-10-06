# Use Ubuntu 22.04 LTS as base image
FROM ubuntu:22.04

# Set maintainer label
LABEL maintainer="your-email@example.com"

# Update package list and install basic utilities
RUN apt-get update && \
    apt-get install -y \
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create a non-root user
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Copy inference.py to /app directory
COPY inference.py /app/

# Default command
CMD ["bash"]
