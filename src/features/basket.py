import pandas as pd
import numpy as np
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.io import load_raw_data, save_processed_data

def clean_transactions():
    print("   [Basket] Loading raw transactions...")
    # Đọc dữ liệu thô (giả định tên file chuẩn)
    tx = load_raw_data("transaction_data.csv")
    prod = load_raw_data("product.csv")
    
    # Merge Product info
    tx = tx.merge(prod[["product_id", "department", "commodity_desc"]], on="product_id", how="left")
    
    # Data Cleaning cơ bản
    tx = tx[(tx["quantity"] > 0) & (tx["sales_value"] >= 0)].copy()
    tx["unit_price"] = tx["sales_value"] / tx["quantity"]
    
    # Xử lý Outlier giá
    lo, hi = tx["unit_price"].quantile([0.005, 0.995])
    tx["unit_price"] = tx["unit_price"].clip(lo, hi)
    
    print(f"   [Basket] Saving cleaned transactions: {tx.shape} rows")
    save_processed_data(tx, "tx_clean.parquet")
    return tx