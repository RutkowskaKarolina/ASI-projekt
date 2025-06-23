# Use slim Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Pre-copy requirements for caching
COPY requirements.txt .

# Install pip packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Optional: install explicitly versioned extras
RUN pip install --no-cache-dir \
    streamlit==1.28.0 \
    torch==2.3.0 \
    torchvision==0.18.0 \
    lightning==2.2.3 \
    pillow==10.0.0 \
    opencv-python==4.8.1.78


# Copy project
COPY . .

# Ensure required folders exist (optional, can also be in app logic)
RUN mkdir -p \
    data/01_raw/indian_food \
    data/02_intermediate \
    data/06_models \
    data/08_reporting \
    models

# Add non-root user
RUN useradd -m streamlit && \
    chown -R streamlit:streamlit /app
USER streamlit

# Expose port
EXPOSE 8501

# Healthcheck (useful on Hugging Face too)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
