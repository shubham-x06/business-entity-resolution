FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/code/business_entity_resolution/src

# Set working directory to repository root
WORKDIR /app

# Install system dependencies and uv package manager
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Copy dependencies manifest and install via uv into system environment
COPY code/business_entity_resolution/requirements.txt /app/requirements.txt
RUN uv pip install --system -r /app/requirements.txt

# Copy the full repository into the container
COPY . /app

# Pipeline commands are invoked via docker-compose run or explicit execution
CMD ["bash"]
