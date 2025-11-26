import os
from pathlib import Path

# Định vị thư mục gốc (Project Root)
# Logic: file này ở src/config.py -> cha là src -> cha của src là ROOT
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent  # D:\STAT3013.Q12_Group01

# Định nghĩa các thư mục dữ liệu
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "features"

# Định nghĩa thư mục kết quả
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"         
MODELS_DIR = PROJECT_ROOT / "src" / "models"

for d in [PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42