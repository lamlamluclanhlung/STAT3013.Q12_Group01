# From Coupon to Comeback

### Promotion Analytics & Repeat Purchase Prediction for Retail Customers

**Course:** STAT3013 – Application of Statistics & ML/DL in Economics, Business, and Healthcare  
**Class:** Q12 • **Group:** 01 • **Repo:** `STAT3013.Q12_Group01`

---

## 1. Project Overview

This project analyses how price promotions and coupons affect customer repeat purchases in a retail setting.  
We use a real-world–style transaction dataset with basket-level information to:

- Measure the **causal/associational impact** of promotions on **repeat within 30 days**.
- Estimate **price elasticity** at the product category level.
- Build **time-series forecasting models** for weekly category sales.
- Develop **classification and uplift models** to target customers for future campaigns.

The pipeline combines classical statistics and modern ML/DL:

- Descriptive & inferential statistics (t-tests, chi-square, effect sizes).
- Regression for price elasticity.
- ARIMA & Temporal Fusion Transformer (TFT) for forecasting.
- Logistic regression & FT-Transformer + uplift modeling for campaign targeting.

---

## 2. Repository Structure

```text
STAT3013.Q12_Group01/
├── notebooks/
│   ├── 00_prep_features.ipynb              # Load & clean raw data, save tx_clean.parquet
│   ├── 01_eda_and_cleaning.ipynb          # EDA & data quality checks
│   ├── 02_build_features_repeat30d.ipynb  # Create repeat@30d label & customer features
│   ├── 03_build_weekly_category.ipynb     # Weekly category aggregation for forecasting
│   ├── 04_inference_tests.ipynb           # Chi-square & t-test on promotion vs repeat
│   ├── 05_elasticity_models.ipynb         # Price elasticity models
│   ├── 06_forecasting_baseline_and_tft.ipynb  # ARIMA/Prophet/TFT forecasting
│   ├── 07_repeat_prediction_and_uplift.ipynb  # Repeat prediction & uplift modeling
│   └── run_all_notebooks.py     # Runs 00–07 sequentially via nbconvert
│
├── src/
│   ├── config.json          # Project paths (relative to repo root)
│   ├── config.py            # Python config helpers (PROJECT_ROOT, DIRs)
│   ├── io.py                # Load/save utilities for raw & feature data
│   ├── eval.py              # Metric helpers (regression & classification)
│   ├── plots.py             # Common plotting utilities (calibration, ROI frontier)
│   ├── utils.py             # Seeding, helper functions
│   ├── pipeline.py          # Orchestrates feature + model pipeline from src.models/src.features
│   ├── features/
│   │   ├── __init__.py
│   │   ├── basket.py        # Transaction-level cleaning and enrichment
│   │   ├── repeat30d.py     # Label engineering for repeat@30d & customer RFM-like features
│   │   └── weekly_category.py # Weekly category aggregation for forecasting
│   └── models/
│       ├── __init__.py
│       ├── elasticity.py    # Elasticity regression & summaries
│       ├── forecasting.py   # ARIMA/Prophet/TFT wrappers for weekly_category
│       ├── inference.py     # Chi-square, t-tests, effect sizes
│       └── repeat_uplift.py # Repeat prediction & uplift modeling utilities
│
├── results/                 # Metrics & tables (CSV/JSON/Parquet) produced by notebooks
├── figures/                 # Plots used in the report / IEEE paper
├── data/                    # (Optional) Raw data, if included; otherwise kept on cloud
├── requirements.txt         # Python dependencies (see Environment section)
├── README.md                # This file
└── LICENSE                  # Project license (educational use / MIT)
```

> **Note:** The `src/` directory is the “engine room” of the project.  
> It contains reusable source code (config, IO, feature engineering, models, evaluation, plotting)  
> that is used across all notebooks and helps keep the whole pipeline consistent and maintainable.

---

## 3. Data

We use the **dunnhumby “The Complete Journey”** retail transaction dataset.

- **Source:** dunnhumby – Customer Data Science & Consultancy
- **Public sample dataset:** _The Complete Journey_
- **URL:** https://www.dunnhumby.com/source-files/

This dataset contains household-level grocery transactions, including:

- Basket-level sales with product, price, and promotion information
- Loyalty card identifiers (household keys)
- Promotion flags (e.g., display, mailer, price reduction)
- Temporal information (day, week, etc.)

In this project, we:

- Clean and preprocess the raw dunnhumby files into:
  - `tx_clean.parquet` – cleaned transactions
  - `repeat30d.parquet` – customer-period features with the `repeat30d` label and treatment flag
  - `weekly_category.parquet` – weekly category-level aggregates for forecasting
- Use only derived/aggregated features for modelling and reporting.

> **Note:** The original dunnhumby “The Complete Journey” dataset is owned by dunnhumby.  
> It is used here strictly for **educational and non-commercial purposes** in the STAT3013 course.

---

## 4. Environment & Setup

We recommend using **conda** with Python 3.10+.

### 4.1. Create environment

```bash
# Clone repository
git clone https://github.com/<your-account>/STAT3013.Q12_Group01.git
cd STAT3013.Q12_Group01

# Using requirements.txt
conda create -n stat_env python=3.11
conda activate stat_env
pip install -r requirements.txt
```

The main libraries include:

- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
- `statsmodels`, `lightgbm` / `xgboost` (optional)
- `torch` (+ packages needed for TFT, if used)
- `jupyter`, `nbconvert`

### 4.2. Config paths

`src/config.json` stores **relative paths**:

```json
{
  "RAW": "data/raw",
  "FEATURES": "features",
  "RESULTS": "results",
  "FIGURES": "figures",
  "SEED": 42
}
```

If you keep the default project structure, **no change is required**.  
If you move folders, adjust `config.json` accordingly.

---

## 5. How to Run the Pipeline

From the `notebooks/` directory:

```bash
cd notebooks
python ../run_all_notebooks.py
```

This script will:

1. Execute all notebooks 00–07 using `nbconvert`.
2. Save intermediate features to `features/`.
3. Write experiment outputs to `results/` and plots to `figures/`.

If any notebook fails, the script will stop and print the last error.

---

## 6. Demo

The demo video and main deliverables are hosted on Google Drive:

- **Google Drive link:** https://drive.google.com/drive/folders/1O27ajaiYyqt2la9nYMrSCNKiE2JK8e9j

The demo video briefly walks through:

- How to run the project (environment, notebooks, and `run_all_notebooks.py`).
- Which **outputs** are generated in the `features/`, `results/`, and `figures/` folders.
- Key **plots and tables** (inference tests, elasticity, forecasting, uplift).
- The main **insights and conclusions** drawn from the results (promotion impact, repeat behavior, and targeting strategy).

---

## 7. License & Academic Integrity

This repository is created for the **STAT3013 course at UIT** and is intended for **educational and non-commercial use only**.

- You may refer to the code and methodology for learning and research purposes.
- Please do **not** reuse this project verbatim for course submissions in future semesters.
- All external libraries and datasets are used under their respective licenses.

---

## 8. Contact

For questions about this project:

- **Course:** STAT3013 – Application of Statistics & ML/DL in Economics, Business, and Healthcare
- **Class:** Q12 – Group 01
- **Supervisor:** Course instructor, STAT3013, UIT
- **Student contact:** Nguyễn Chí Lâm
