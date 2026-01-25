@echo off
echo Creating virtual environment...
python -m venv venv

echo Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Setup complete! Use 'run.bat --help'
pause