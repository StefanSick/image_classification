#!/bin/bash
set -e

echo "--- Validating Conda Installation ---"

if ! command -v conda &> /dev/null; then
    echo "Conda is required but was not found in PATH."
    exit 1
fi

echo "Conda detected: $(conda --version)"

echo "--- Validating Tools Environment ---"

if ! conda env list | awk '{print $1}' | grep -qx "conda-tools"; then
    echo "Required environment 'conda-tools' not found."
    echo "Run the following command once:"
    echo ""
    echo "  conda create -n conda-tools -c conda-forge mamba -y"
    echo ""
    exit 1
fi

echo "conda-tools environment found."

if [ ! -f environment.yml ]; then
    echo "environment.yml not found in the current directory."
    exit 1
fi

echo "--- Creating / Updating Project Environment ---"

# Extract the environment name from environment.yml
ENV_NAME=$(grep "name:" environment.yml | awk '{print $2}')

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' not found. Creating..."
    conda run -n conda-tools mamba env create -f environment.yml
else
    echo "Environment '$ENV_NAME' exists. Updating..."
    conda run -n conda-tools mamba env update -f environment.yml --prune
fi