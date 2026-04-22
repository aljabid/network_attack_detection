# Multi-stage Dockerfile for AI-Based Network Attack Detection System
# Production-ready container with optimized size and security

# Stage 1: Builder
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim

# Set labels
LABEL maintainer="your.email@example.com"
LABEL description="AI-Based Network Attack Detection System"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app \
    LOG_DIR=/app/logs \
    DATA_DIR=/app/data \
    MODEL_DIR=/app/data/models

# Create non-root user
RUN groupadd -r detector && \
    useradd -r -g detector -d /app -s /sbin/nologin detector

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR ${APP_HOME}

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY --chown=detector:detector . .

# Create necessary directories
RUN mkdir -p ${LOG_DIR} ${DATA_DIR} ${MODEL_DIR} && \
    chown -R detector:detector ${APP_HOME}

# Make scripts executable
RUN chmod +x setup.sh

# Update Python path
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=${APP_HOME}/src:$PYTHONPATH

# Switch to non-root user
USER detector

# Expose port (if using web interface)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "src/realtime_detection.py"]
