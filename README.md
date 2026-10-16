# Concrete Compressive Strength Project 1

## Files to upload to GitHub

For a reproducible code repository, upload these required files:

- `analyze_concrete.py` — analysis implementation, including custom batch gradient descent.
- `requirements.md` — pinned dependency versions and a copyable installation command.
- `README.md` — project overview and run instructions.
- `Concrete_Data.xls` — input data used by the script.
- `Concrete_Readme.txt` — dataset field descriptions and units.

The `results/` folder is optional but useful if you want the repository to include the CSV tables and PNG figures used in the reports. Do not upload virtual environments, temporary render folders, or editor caches.

## Reproduce the analysis

Use Python 3.12. Install the pinned packages using the command in [`requirements.md`](requirements.md). Then run this command from the project folder:

```bash
python analyze_concrete.py --data Concrete_Data.xls --output-dir results
```

The script preserves the original row order and reserves observations 501–630 (one-based) for testing. Preprocessing is fit on the 900 training records only. It writes CSV result tables and PNG figures to the requested output directory. The included `results/` folder contains the outputs used in both reports.

## Dataset credit

Yeh, I.-C. (1998). *Concrete Compressive Strength* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PK67.
