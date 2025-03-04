### Install micromamba from https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html (04/03/2025) ###

# Install bzip2 if not installed
apt install bzip2

# Get the archive and extract it in bin/micromamba
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

# Init the shell
./bin/micromamba shell init -s bash -r ~/micromamba

# Apply the modifications on the current shell
source ~/.bashrc

# Create the virtual environment with the required packages (pygimli, ipykernel for notebooks, json5 for configuration files)
micromamba create -n venv python=3.11 gimli::pygimli ipykernel json5 -c conda-forge

# Activate the environment
micromamba activate venv

# Update pip
pip3 install --upgrade pip

# Install PyTorch with GPU support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126