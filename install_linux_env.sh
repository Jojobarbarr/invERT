#!/bin/bash
set -e  # Stop the script if any command fails

### Install micromamba from https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html (04/03/2025) ###

sudo apt update && sudo apt upgrade
if ! command -v bzip2 &> /dev/null; then
	echo "bzip2 not found. Installing..."
	sudo apt install -y bzip2
fi

# Get the archive and extract it in bin/micromamba
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

export MAMBA_ROOT_PREFIX='/root/micromamba'
eval "$(./bin/micromamba shell hook -s posix)"

# Init the shell
./bin/micromamba shell init -s bash -r ~/micromamba
source ~/.bashrc

# Create the virtual environment with the required packages (pygimli, ipykernel for notebooks, json5 for configuration files)
micromamba create -n venv -y python=3.11 gimli::pygimli ipykernel json5 -c conda-forge

# Activate the environment
micromamba activate venv

# Update pip
pip3 install --upgrade pip

# Install PyTorch with GPU support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo "Installation complete. Run 'micromamba activate venv' to use the environment."