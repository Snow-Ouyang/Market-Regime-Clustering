# Raw Data

This folder contains the public raw inputs used in the project.

- `macro/`: monthly and quarterly macro series downloaded from FRED and saved as CSV.
- `assets/`: market series used for the narrative and asset-mapping section.

Notes:

- Most macro raw files are directly reproducible from FRED.
- `gold.csv`, `oil.csv`, and `GSPC_yfinance_daily.csv` are included here as flat files.
- The bond proxy (`VUSTX`) is downloaded inside the reporting script via `yfinance` and is therefore not stored as a raw CSV in the repository.

