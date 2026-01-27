#!/bin/bash

# Activate virtual environment
source venv/bin/activate

while true; do
    clear

    echo "=== DATASET ==="
    echo "1. Fashion-MNIST"
    echo "2. CIFAR-10"
    echo -n "Dataset (1-2): "
    read ds

    case "$ds" in
        1) dataset="fashion_mnist" ;;
        2) dataset="cifar10" ;;
        *) continue ;;
    esac

    clear

    echo "=== MODEL ==="
    echo "1. CNN"
    echo "2. RNN"
    echo "3. ViT"
    echo "4. HIST"
    echo "5. SIFT"
    echo -n "Model (1-5): "
    read mt

    case "$mt" in
        1) model_type="cnn" ;;
        2) model_type="rnn" ;;
        3) model_type="vit" ;;
        4) model_type="HIST" ;;
        5) model_type="SIFT" ;;
        *) continue ;;
    esac

    clear

    echo "=== MODE ==="
    echo "1. Train"
    echo "2. Test"
    echo -n "Mode (1-2): "
    read md

    case "$md" in
        1) mode="train" ;;
        2) mode="test" ;;
        *) continue ;;
    esac

    # Default params; only ask in train + cnn/rnn/vit
    epochs=""
    batch_size=""

    if [ "$mode" = "train" ]; then
        if [ "$model_type" = "cnn" ] || [ "$model_type" = "rnn" ] || [ "$model_type" = "vit" ]; then
            clear
            echo "=== PARAMS (Enter or press Enter for default) ==="

            echo -n "Epochs [15]: "
            read in_epochs
            if [ -z "$in_epochs" ]; then
                epochs="15"
            else
                epochs="$in_epochs"
            fi

            echo -n "Batch size [64]: "
            read in_batch
            if [ -z "$in_batch" ]; then
                batch_size="64"
            else
                batch_size="$in_batch"
            fi
        fi
    fi

    clear

    echo "=== CONFIRM ==="
    echo "Dataset: $dataset"
    echo "Model:   $model_type"
    echo "Mode:    $mode"

    if [ "$mode" = "train" ]; then
        if [ "$model_type" = "cnn" ] || [ "$model_type" = "rnn" ] || [ "$model_type" = "vit" ]; then
            echo "Epochs:  $epochs"
            echo "Batch:   $batch_size"
        fi
    fi

    echo
    echo -n "Run? (Y/N): "
    read confirm

    if [ "$(echo "$confirm" | tr '[:upper:]' '[:lower:]')" != "y" ]; then
        continue
    fi

    echo "Running command..."

    # Use classifierSL.py for HIST/SIFT, classifierDL.py for cnn/rnn/vit
    if [ "$model_type" = "HIST" ]; then
        python src/classifierSL.py \
            --dataset "$dataset" \
            --model_type "$model_type" \
            --mode "$mode"
    elif [ "$model_type" = "SIFT" ]; then
        python src/classifierSL.py \
            --dataset "$dataset" \
            --model_type "$model_type" \
            --mode "$mode"
    else
        # cnn, rnn, vit -> classifierDL.py
        if [ "$mode" = "train" ]; then
            python src/classifierDL.py \
                --dataset "$dataset" \
                --model_type "$model_type" \
                --mode "$mode" \
                --epochs "$epochs" \
                --batch-size "$batch_size"
        else
            python src/classifierDL.py \
                --dataset "$dataset" \
                --model_type "$model_type" \
                --mode "$mode"
        fi
    fi

    echo
    read -s -n1 -p "Press Enter to run another experiment..."
done
