#!/bin/bash
set -e  # Exit immediately if a command fails

echo "--- Starting Conda Setup ---"

# Create or Update the environment

echo "Installing/Updating libraries from environment.yml..."
conda env update -f environment.yml --prune

# Create the models directory (required for your Python script)
if [ ! -d "models" ]; then
    mkdir models
    echo "Created /models directory."
fi

echo "--- Setup Complete! ---"
