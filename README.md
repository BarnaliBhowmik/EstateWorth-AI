# Housing Regression MLE

Overview
- Time-series housing price regression and ML pipeline.

Windows development instructions

Prerequisites
- Python 3.11
- Git
- (Optional) Docker Desktop with WSL2 backend for container builds

Setup (PowerShell)
```powershell
# create virtualenv
python -m venv venv
# activate
& .\venv\Scripts\Activate.ps1
# upgrade build tools and pip
python -m pip install --upgrade pip setuptools wheel
# install package and dev deps
pip install -e .[dev]
```

Setup (Command Prompt)
```cmd
python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip install -e .[dev]
```

Running locally
- Run API (development):
```powershell
uvicorn src.api.main:app --reload
```

- Run Streamlit app:
```powershell
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

Docker notes (Windows)
- Use Docker Desktop with WSL2 backend for best compatibility. Build Streamlit image:
```powershell
docker build -f Dockerfile.streamlit -t housing-streamlit .
docker run -p 8501:8501 housing-streamlit
```

Troubleshooting
- If you see long path / permission issues on Windows with Docker, enable "Use the WSL 2 based engine" in Docker Desktop and share resources.
- If Python package installation fails due to PEP 517 build isolation, ensure `pip`, `setuptools`, and `wheel` are upgraded (see setup steps).

If you want, I can add CI updates or a Windows-specific GitHub Actions runner next.
