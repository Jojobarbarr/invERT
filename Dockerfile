# Use miniconda as base image
FROM continuumio/miniconda3:latest

# Update and install build-essential and curl
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# Create a conda environment with your desired Python version and install pygimli
RUN conda create -n venv python=3.11.11 -y && \
    conda install -n venv -c conda-forge -y gimli::pygimli && \
    conda clean -afy

# Set the default shell to bash
SHELL ["/bin/bash", "-c"]

# Activate the environment and install pip dependencies
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate venv && pip install --upgrade pip

ARG USE_GPU=0
ENV USE_GPU=${USE_GPU}

RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate venv && \
    if [ "$USE_GPU" -eq "1" ]; then \
      echo "Installation de PyTorch avec support GPU"; \
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126; \
    else \
      echo "Installation de PyTorch (CPU uniquement)"; \
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Installation des autres dépendances de votre projet (à adapter dans requirements.txt)
COPY requirements.txt requirements.txt
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate venv && pip install --no-cache-dir -r requirements.txt

# Ensure conda is available in the interactive shell session
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate venv" >> ~/.bashrc

# Launch interactive shell in venv by default
CMD ["bash", "-i"]

