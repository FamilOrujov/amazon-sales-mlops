import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    print("Starting MLflow UI...")
    print("Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop")
    print()
    
    
    cmd = [
        sys.executable, "-m", "mlflow", "ui",
        "--host", "0.0.0.0",
        "--port", "5000",
    ]
    
    subprocess.run(cmd, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
