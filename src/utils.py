import os
import random
import numpy as np
import pandas as pd
import warnings

def seed_all(seed=42):
    """Cố định Random Seed cho cả Numpy, Python và PyTorch"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Bổ sung cho Deep Learning
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def lower_columns(df: pd.DataFrame):
    """Chuẩn hóa tên cột: chữ thường + bỏ khoảng trắng"""
    df.columns = [str(c).lower().strip() for c in df.columns]
    return df

def safe_div(a, b, fill=0.0):
    """Chia an toàn, tránh lỗi chia cho 0"""
    b0 = np.where(b == 0, 1, b)
    out = a / b0
    if np.isscalar(fill):
        out = np.where(np.isfinite(out), out, fill)
    return out

def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.where(y_true == 0, 1, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom))

warnings.filterwarnings("ignore")