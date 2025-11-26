import pandas as pd
from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from .utils import lower_columns

def load_raw_data(filename: str, **kwargs):
    """Đọc 1 file CSV từ folder data/raw"""
    path = RAW_DATA_DIR / filename
    try:
        df = pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        # Fallback nếu file lỗi font
        df = pd.read_csv(path, encoding='latin1', **kwargs)
    return lower_columns(df)

def load_processed_data(filename: str):
    """Đọc file Parquet từ folder features"""
    path = PROCESSED_DATA_DIR / filename
    return pd.read_parquet(path)

def save_processed_data(df: pd.DataFrame, filename: str):
    """Lưu file Parquet vào folder features"""
    path = PROCESSED_DATA_DIR / filename
    # Tạo thư mục cha nếu chưa có
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"--> Saved: {path.name}")
