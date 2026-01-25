# Image Classification: CNN, RNN, and Vision Transformers

This project explores the impact of data representation by comparing traditional computer vision techniques (SIFT) against modern deep learning architectures (CNN, RNN, and ViT).

---

##  Setup & Installation

### 1. Environment Setup
This project uses **Conda**. Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate image_classification
```

### 2. Project Initialization
Run the setup script to prepare the directory structure and fetch binary files via Git LFS.
* **Windows (CMD/PowerShell):** `bash setup.sh`
* **Linux/macOS/Git Bash:** `./setup.sh`

---

## Usage Deep Learning
Use the `run.sh` wrapper to execute training or testing. 

```bash
bash run.sh
```
## Other option:
| Goal | Model Type | Command |
| :--- | :--- | :--- |
| **Train CNN** | `cnn` | `python src/classifier3.py --dataset fashion_mnist --model_type cnn --mode train`
| **Train RNN** | `rnn` | `python src/classifier3.py --dataset cifar10 --model_type rnn --mode train`
| **Train ViT** | `vit` | `python src/classifier3.py --dataset cifar10 --model_type vit --mode train`
| **Test Model** | `any` | `python src/classifier3.py --dataset cifar10 --model_type vit --mode test`

> **Note:** Models are saved in the `/models` directory as `.keras` files. Ensure you have trained a model before attempting to run it in `--mode test`.

---
