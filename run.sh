#!/bin/bash

# Initialize Conda for this script session
# This finds where conda is installed and enables the 'conda' command
source $(conda info --base)/etc/profile.d/conda.sh

# Activate the environment
conda activate image_classification

while true; do
  clear
  echo "=== DATASET ==="
  echo "1. Fashion-MNIST"
  echo "2. CIFAR-10"
  read -p "Dataset (1-2): " ds

  case "$ds" in
    1) dataset="fashion_mnist" ;;
    2) dataset="cifar10" ;;
    *) echo "Invalid choice."; sleep 1; continue ;;
  esac

  while true; do
    clear
    echo "=== MODEL ==="
    echo "1. CNN"
    echo "2. RNN"  
    echo "3. Vision Transformer (ViT)"
    echo "4. HIST (Traditional)"
    echo "5. SIFT (Traditional)"
    read -p "Model (1-5): " mt

    case "$mt" in
      1) model_type="cnn" ;;
      2) model_type="rnn" ;;
      3) model_type="vit" ;;
      4) model_type="hist" ;;
      5) model_type="sift" ;;
      *) echo "Invalid choice."; sleep 1; continue ;;
    esac
    break
  done

  while true; do
    clear
    echo "=== MODE ==="
    echo "1. Train" 
    echo "2. Test"
    read -p "Mode (1-2): " md

    case "$md" in
      1) mode="train" ;;
      2) mode="test" ;;
      *) echo "Invalid choice."; sleep 1; continue ;;
    esac
    break
  done

  if [[ "$mode" == "train" ]]; then
    clear
    echo "=== PARAMS (Enter for default) ==="
    read -p "Epochs [15]: " epochs
    epochs=${epochs:-15}
    read -p "Batch size [64]: " batch_size
    batch_size=${batch_size:-64}
  fi

  clear
  echo "=== CONFIRM ==="
  echo "Dataset: $dataset"
  echo "Model: $model_type" 
  echo "Mode: $mode"
  if [[ "$mode" == "train" ]]; then
    echo "Epochs: $epochs"
    echo "Batch: $batch_size"
  fi
  echo
  read -p "Run? (Y/N): " confirm
  [[ "$confirm" =~ ^[Yy]$ ]] || continue

  echo "Running command..."
  

  if [[ "$mode" == "train" ]]; then
    python src/classifier3.py \
      --dataset "$dataset" \
      --model_type "$model_type" \
      --mode "$mode" \
      --epochs "$epochs" \
      --batch-size "$batch_size"
  else
    python src/classifier3.py \
      --dataset "$dataset" \
      --model_type "$model_type" \
      --mode "$mode"
  fi

  echo
  read -p "Press Enter to run another experiment (or Ctrl+C to exit)..."
done