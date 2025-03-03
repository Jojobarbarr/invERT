# Create a new conda environment "venv" with Python 3.11.11
conda create -n venv python=3.11.11 -y

# Install pygimli from the conda-forge channel using the "gimli::pygimli" specification
conda install -n venv -c conda-forge -y vtk gimli::pygimli

# Load the conda hook to set up conda commands in the session
& "C:\anaconda3\shell\condabin\conda-hook.ps1"

# Activate the conda environment "venv"
conda activate venv

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch and related packages (using the CPU-only wheels from the provided index URL)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install additional dependencies as listed in requirements.txt
python -m pip install -r requirements.txt

# Create a helper script to activate the conda environment later if needed.
Set-Content -Path "activate_conda.ps1" -Value "powershell -ExecutionPolicy ByPass -NoExit -Command '& 'C:\anaconda3\shell\condabin\conda-hook.ps1'; conda activate venv'"
