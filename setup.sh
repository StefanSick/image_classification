#!/bin/bash
set -e

echo "--- Starting Environment Setup ---"

# 1. Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    
    # Download the installer (Linux version)
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    
    # Run installer in batch mode (-b) and update shell profile (-u)
    bash Miniconda3-latest-Linux-x86_64.sh -b -u
    
    # Initialize conda for the current shell session
    source ~/miniconda3/bin/activate
    conda init
    
    echo "Miniconda installed successfully."
else
    echo "Conda is already installed. Skipping installation."
fi

# 2. Install Mamba into the base environment
if ! command -v mamba &> /dev/null; then
    echo "Mamba not found. Installing Mamba via conda-forge..."
    conda install mamba -n base -c conda-forge -y
else
    echo "Mamba is already installed."
fi

# 3. Create or Update the environment using Mamba
echo "Installing/Updating libraries from environment.yml using MAMBA..."
# We use mamba here for the 10x speed boost in dependency solving
mamba env update -f environment.yml --prune

# 4. Create the models directory
if [ ! -d "models" ]; then
    mkdir models
    echo "Created /models directory."
fi

echo "--- Setup Complete! ---"