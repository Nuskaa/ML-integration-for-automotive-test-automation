# DM ML Testing Pipeline

Lightweight pipeline to prioritize automated tests using engineered features and ML models.

## Overview

This repository contains a pipeline that:
- Loads test suites
- Engineers static and optional execution features
- Runs a prioritization model and detects flaky tests
- Outputs a ranked test order and an Excel report

Main entry point: [run_pipeline.py](run_pipeline.py)

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- openpyxl (optional, used to generate the Excel report)

Install dependencies (recommended inside a virtualenv):

```bash
python -m venv .venv
source .venv/Scripts/activate    # PowerShell/Windows: .venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn openpyxl
```

## Usage

Basic run (uses static features):

```bash
python run_pipeline.py --data /path/to/DM_All_TestSuites --output ./outputs
```

With optional execution history and benchmark files:

```bash
python run_pipeline.py --data ./data --history ./history.csv --benchmark ./benchmarks.csv --output ./outputs
```

The script will create the `outputs/` directory (if missing) and write:

- `outputs/prioritized_test_order.csv` — ranked test order for TestRunner
- `outputs/ml_test_report.xlsx` — multi-sheet Excel report (openpyxl required)
- `outputs/models/prioritizer.pkl` — saved prioritization model
- `outputs/models/anomaly_detector.pkl` — saved anomaly detector (if benchmark provided)

If `openpyxl` is not installed, the pipeline will fall back to saving CSV exports for the report.

## Quick Notes for Developers

- Data loader: [data_loader.py](data_loader.py)
- Feature engineering: [feature_engineering.py](feature_engineering.py)
- Models: [ml_models.py](ml_models.py)
- Entry point: [run_pipeline.py](run_pipeline.py)

To run tests or iterate, update the code and re-run the command above. Consider adding a `requirements.txt` or `pyproject.toml` to pin dependencies for reproducible runs.

## Troubleshooting

- "openpyxl not available": install `openpyxl` to enable Excel output.
- Git/GitHub: if you plan to push this repo, initialize Git (`git init`), create a remote on GitHub, then:

```powershell
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/USERNAME/REPO.git
git branch -M main
git push -u origin main
```

Replace the remote URL with your repository's URL. For HTTPS pushes you may need a Personal Access Token.

## License

Add a license file if you intend to publish this repository publicly.
