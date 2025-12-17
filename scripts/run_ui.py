# Streamlit UI
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "src" / "amazon_sales_ml" / "ui" / "app.py"


def main():
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(APP_PATH),
        "--server.port", "8501",
        "--server.address", "localhost",
    ]
    
    print(f"Starting Streamlit UI...")
    print(f"Open http://localhost:8501 in your browser")
    print()
    
    subprocess.run(cmd)


if __name__ == "__main__":
    main()

