# Image Classification: CNN, RNN, and Vision Transformers

This project explores the impact of data representation by comparing shallow learning techniques (3-Channel Histogram and SIFT + BoVW) against modern deep learning architectures (CNN, RNN, and ViT).

---
##  Setup & Installation

### 1. Run setup.bat
This will create a Virtual Environment for python and download all required dependencies and libraries.

### 2. Run run.bat
This will start the graphical CLI from where both datasets can be used on all 5 models for either training or testing, 
either with our saved models (provided separetely in Submission of large additional Files) or with self trained models.

On Windows:

    *   Run:  setup.bat (or double-click it in File Explorer).

    *   Run:  run.bat  

     
On Linux/macOS:

    *   Open your terminal.

    *   Navigate to this folder.
    
    *   Run: 
        chmod +x setup.sh && ./setup_dl.sh
        chmod +x run.sh && ./run_dl.sh


## Usage
* When you launch the script, you will be presented with an interactive menu to configure your run:

* Select Dataset: Choose between Fashion-MNIST or CIFAR-10.

* Select Model: Choose the architecture (HIST, SIFT, CNN, RNN, or ViT).

* Select Mode: - Train: You will be prompted for Epochs and Batch Size (DL models only).

* Test: The script skips parameter input and runs evaluation immediately.

## Project Structure
* src/classifierDL.py — The Python file for all DL models (CNN, RNN, ViT).
* src/classifierSL.py — The Python file for Histogram and SIFT.
* run.bat / run.sh — Interactive wrappers for Windows and Unix.
* requirements.txt — Lists of libraries loaded into the Virtual Environment.




