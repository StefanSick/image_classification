@echo off
setlocal enabledelayedexpansion

call activate image_classification_new

:main_menu
:: Clear variables so old choices don't "leak" into new runs
set "dataset="
set "model_type="
set "mode="
set "epochs="
set "batch_size="

cls
echo === DATASET ===
echo 1. Fashion-MNIST
echo 2. CIFAR-10
set /p ds="Dataset (1-2): "

if "%ds%"=="1" (set "dataset=fashion_mnist") else if "%ds%"=="2" (set "dataset=cifar10") else (goto main_menu)

:model_menu
cls
echo === MODEL ===
echo 1. CNN
echo 2. RNN
echo 3. Vision Transformer (ViT)
echo 4. HIST
echo 5. SIFT
set /p mt="Model (1-5): "

if "%mt%"=="1" (set "model_type=cnn") else if "%mt%"=="2" (set "model_type=rnn") else if "%mt%"=="3" (set "model_type=vit") else if "%mt%"=="4" (set "model_type=HIST") else if "%mt%"=="5" (set "model_type=SIFT") else (goto model_menu)

:mode_menu
cls
echo === MODE ===
echo 1. Train
echo 2. Test
set /p md="Mode (1-2): "
if "%md%"=="1" set mode=train
if "%md%"=="2" set mode=test
if not defined mode goto mode_menu

REM Only ask for params if TRAIN AND (cnn, rnn, or vit)
if "%mode%"=="train" (
    if "%model_type%"=="cnn" goto params
    if "%model_type%"=="rnn" goto params
    if "%model_type%"=="vit" goto params
)
goto confirm

:params
cls
echo === PARAMS (Enter for default) ===
set "epochs=15"
set /p input_epochs="Epochs [15]: "
if not "!input_epochs!"=="" set "epochs=!input_epochs!"

set "batch_size=64"
set /p input_batch="Batch size [64]: "
if not "!input_batch!"=="" set "batch_size=!input_batch!"

:confirm
cls
echo === CONFIRM ===
echo Dataset: %dataset%
echo Model:   %model_type%
echo Mode:    %mode%
if "%mode%"=="train" (
    if "%model_type%"=="cnn" (
        echo Epochs: %epochs%
        echo Batch: %batch_size%
    ) else if "%model_type%"=="rnn" (
        echo Epochs: %epochs%
        echo Batch: %batch_size%
    ) else if "%model_type%"=="vit" (
        echo Epochs: %epochs%
        echo Batch: %batch_size%
    ) 
)
REM EXECUTE
echo.
set /p confirm="Run? (Y/N): "
if /i not "%confirm%"=="Y" goto main_menu

echo Running command...

REM Use classifier.py for HIST/SIFT, classifier3.py for cnn/rnn/vit
if "%model_type%"=="HIST" (
        python src\classifier.py ^
        --dataset %dataset% ^
        --model_type %model_type% ^
        --mode %mode%

) else if "%model_type%"=="SIFT" (
        python src\classifier.py ^
          --dataset %dataset% ^
          --model_type %model_type% ^
          --mode %mode%
) else (
    REM cnn, rnn, vit use classifier3.py
    if "%mode%"=="train" (
        python src\classifier3.py ^
          --dataset %dataset% ^
          --model_type %model_type% ^
          --mode %mode% ^
          --epochs %epochs% ^
          --batch-size %batch_size%
    ) else (
        python src\classifier3.py ^
          --dataset %dataset% ^
          --model_type %model_type% ^
          --mode %mode%
    )
)

echo.
pause
set /p continue="Press Enter to run another experiment..."
goto main_menu