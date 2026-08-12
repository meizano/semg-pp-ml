# Windows User Guide & Setup Tips

Welcome to the sEMG Preprocessing & Machine Learning Platform for Windows!

## 1. Prerequisites on Windows
- **Python 3.10+**: Download and install from [python.org](https://www.python.org/). Ensure you check **"Add Python to PATH"** during installation.
- **Git for Windows**: Download from [git-scm.com](https://git-scm.com/).

## 2. Opening PowerShell & Setting Execution Policy
If PowerShell blocks virtual environment activation (`Activate.ps1`), run this command in PowerShell as Administrator or for your user:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 3. Creating Virtual Environment & Installing
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

## 4. Running the Web App on Windows
```powershell
streamlit run app.py
```
This will launch the application in your browser at `http://localhost:8501`.

## 5. Changing Language
To switch between English and Bahasa Indonesia, open `config/locales/lang_eng.json` or `config/locales/lang_id.json`, or toggle language directly in the Streamlit sidebar.
