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
set /p mt="Model (1-3): "

if "%mt%"=="1" (set "model_type=cnn") else if "%mt%"=="2" (set "model_type=rnn") else if "%mt%"=="3" (set "model_type=vit") else (goto model_menu)

:mode_menu
cls
echo === MODE ===
echo 1. Train
echo 2. Test
set /p md="Mode (1-2): "

if "%md%"=="1" (
    set "mode=train"
) else if "%md%"=="2" (
    set "mode=test"
    :: SKIP the params and go straight to confirmation
    goto confirm 
) else (
    goto mode_menu
)

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
    echo Epochs:  %epochs%
    echo Batch:   %batch_size%
)
echo.
set /p confirm="Run? (Y/N): "
if /i not "%confirm%"=="Y" goto main_menu

echo Running src/classifier3.py...

if "%mode%"=="train" (
    python src/classifier3.py --dataset "%dataset%" --model_type "%model_type%" --mode "%mode%" --epochs "%epochs%" --batch-size "%batch_size%"
) else (
    python src/classifier3.py --dataset "%dataset%" --model_type "%model_type%" --mode "%mode%"
)

echo.
pause
goto main_menu