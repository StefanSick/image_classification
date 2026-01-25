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



## Usage
* When you launch the script, you will be presented with an interactive menu to configure your run:

* Select Dataset: Choose between Fashion-MNIST or CIFAR-10.

* Select Model: Choose the architecture (CNN, RNN, or ViT).

* Select Mode: - Train: You will be prompted for Epochs and Batch Size.

* Test: The script skips parameter input and runs evaluation immediately.

## Project Structure
* src/classifier3.py — The Python logic for all DL models.
* src/classifier.py — The Python logic for HIST and SIFT.
* run.bat / run.sh — Interactive wrappers for Windows and Unix.
* environment.yml — Conda environment configuration.

