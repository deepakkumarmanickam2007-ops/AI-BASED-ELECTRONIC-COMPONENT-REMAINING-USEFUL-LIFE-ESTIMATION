"""One-command reproducible pipeline: dataset -> training -> evaluation."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/"src"))
from generate_dataset import generate_dataset
from train import train_project

def main():
    data_path = ROOT/"data"/"component_aging.csv"
    if not data_path.exists():
        generate_dataset(data_path)
    else:
        print(f"Using bundled dataset: {data_path}")
    train_project(data_path, ROOT/"outputs")
    print("\nDone. Review README.md and docs/project_report.md.")

if __name__ == "__main__":
    main()
