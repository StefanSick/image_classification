# Image Classification: CNN, RNN, and Vision Transformers

This project explores the impact of data representation by comparing traditional computer vision techniques (SIFT) against modern deep learning architectures (CNN, RNN, and ViT).

---
##  Setup & Installation

### 1. Prerequisite: Conda
This project requires **Conda**. If you don't have it, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### 2. Prepare the Environment
We use a dedicated helper environment to handle fast package installation. Run this command **once** in your terminal/prompt:

```bash
conda create -n conda-tools -c conda-forge mamba -y
```

### 3. Initialize & Launch
The project uses automation scripts to check for dependencies, update your environment, and launch the interactive menu.

On Windows:

    *   Open Anaconda Prompt.

    *   Navigate to this folder.

    *   Run: 
        setup_dl.bat
        run_dl.bat  (or double-click it in File Explorer).

On Linux/macOS:

    *   Open your terminal.

    *   Navigate to this folder.
    
    *   Run: 
        chmod +x setup_dl.sh && ./setup_dl.sh
        chmod +x run_dl.sh && ./run_dl.sh

##  Alternative: Python VENV

### 1. Run VENV_setup.bat
This will create a venve and download all required dependencies and libraries

### 2. Run VENV_run.bat
This will start the graphical CLI from where both datasets can be used on all 5 models for either training or testing, 
either with our saved models (provided separetely in Submission of large additional Files) or with self trained models.

## Usage
* When you launch the script, you will be presented with an interactive menu to configure your run:

* Select Dataset: Choose between Fashion-MNIST or CIFAR-10.

* Select Model: Choose the architecture (HIST, SIFT, CNN, RNN, or ViT).

* Select Mode: - Train: You will be prompted for Epochs and Batch Size (DL models only).

* Test: The script skips parameter input and runs evaluation immediately.

## Project Structure
* src/classifier3.py — The Python file for all DL models (CNN, RNN, ViT).
* src/classifier.py — The Python file for Histogramm and SIFT.
* run.bat / run.sh — Interactive wrappers for Windows and Unix.
* environment.yml — Conda environment configuration.

