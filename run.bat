@echo off
call venv\Scripts\activate.bat

:dataset
cls
echo === DATASET ===
echo 1. Fashion-MNIST
echo 2. CIFAR-10
set "dataset="
set /p ds="Dataset (1-2): "
if "%ds%"=="1" set dataset=fashion_mnist
if "%ds%"=="2" set dataset=cifar10
if not defined dataset goto dataset

:model
cls
echo === MODEL ===
echo 1. CNN
echo 2. RNN
echo 3. ViT
echo 4. HIST
echo 5. SIFT
set "model_type="
set /p mt="Model (1-5): "
if "%mt%"=="1" set model_type=cnn
if "%mt%"=="2" set model_type=rnn
if "%mt%"=="3" set model_type=vit
if "%mt%"=="4" set model_type=HIST
if "%mt%"=="5" set model_type=SIFT
if not defined model_type goto model

:mode
cls
echo === MODE ===
echo 1. Train
echo 2. Test
set "mode="
set /p md="Mode (1-2): "
if "%md%"=="1" set mode=train
if "%md%"=="2" set mode=test
if not defined mode goto mode

REM Only ask for params if TRAIN AND (cnn, rnn, or vit)
if "%mode%"=="train" (
    if "%model_type%"=="cnn" goto params
    if "%model_type%"=="rnn" goto params
    if "%model_type%"=="vit" goto params
)
goto confirm

:params
cls
echo === PARAMS (Enter or press Enter for default) ===
set "epochs="
set "batch_size="
set /p epochs="Epochs [15]: "
if "%epochs%"=="" set epochs=15
set /p batch_size="Batch size [64]: "
if "%batch_size%"=="" set batch_size=64

:confirm
cls
echo === CONFIRM ===
echo Dataset: %dataset%
echo Model: %model_type%
echo Mode: %mode%
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
echo.
set /p confirm="Run? (Y/N): "
if /i not "%confirm%"=="Y" goto dataset

REM EXECUTE
echo Running command...

REM Use classifierSL.py for HIST/SIFT, classifierDL.py for cnn/rnn/vit
if "%model_type%"=="HIST" (
        python src\classifierSL.py ^
        --dataset %dataset% ^
        --model_type %model_type% ^
        --mode %mode%

) else if "%model_type%"=="SIFT" (
        python src\classifierSL.py ^
          --dataset %dataset% ^
          --model_type %model_type% ^
          --mode %mode%
) else (
    REM cnn, rnn, vit use classifierDL.py
    if "%mode%"=="train" (
        python src\classifierDL.py ^
          --dataset %dataset% ^
          --model_type %model_type% ^
          --mode %mode% ^
          --epochs %epochs% ^
          --batch-size %batch_size%
    ) else (
        python src\classifierDL.py ^
          --dataset %dataset% ^
          --model_type %model_type% ^
          --mode %mode%
    )
)

echo.
set /p continue="Press Enter to run another experiment..."
goto dataset
