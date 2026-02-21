FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0;7.0;8.0;8.6"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV MLFLOW_TRACKING_URI="http://0.0.0.0:5000"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrading pip
RUN python -m pip install --upgrade pip

# Installing required Python dependencies
RUN pip install --no-cache-dir \
    mlflow \
    matplotlib \
    numpy \
    pandas \
    scikit-learn \
    tokenizers \
    tqdm \
    torchsummary \
    tensorboard

# Expose MLflow UI port
EXPOSE 5000

# Setting the working directory
WORKDIR /workspace

# To Ensure MLflow logs persist
VOLUME /workspace/mlruns

# Starting MLflow tracking server in background, then run bash
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "/workspace/mlruns"]
