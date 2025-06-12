@echo off 
echo Setting up VISIT-Museum-Tracker... 
 
:: Create virtual environment 
python -m venv venv 
call venv\Scripts\activate 
 
:: Install dependencies 
pip install -r requirements.txt 
 
echo Setup complete! 
echo To start the application, run: python -m src.main 
