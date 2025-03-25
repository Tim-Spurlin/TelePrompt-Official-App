@echo off
cd /d "C:\Users\timsp\OneDrive\Desktop\TelePrompt v5\teleprompt"

:: Activate the virtual environment
call .\.venv\Scripts\activate.bat

:: Run the Python script
python main.py

:: Keep the window open in case of errors
pause