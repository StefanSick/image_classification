#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv venv

echo "Installing dependencies..."
source venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete! Use 'run_experiment.sh'"
read -p "Press Enter to continue..."
