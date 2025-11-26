import pandas as pd
import numpy as np
from src.io import load_processed_data, save_processed_data

def create_repeat_target():
    print("   [Repeat30d] Loading cleaned transactions...")
    tx = load_processed_data("tx_clean.parquet")
    
    # Gom nhóm theo Hóa đơn (Basket)
    orders = tx.groupby(["household_key", "basket_id", "day"], as_index=False).agg(
        basket_value=("sales_value", "sum"),
        basket_qty=("quantity", "sum"),
        n_items=("product_id", "nunique")
    )
    orders = orders.sort_values(["household_key", "day"])
    
    # Tạo biến Repeat trong 30 ngày (Target)
    orders["next_day"] = orders.groupby("household_key")["day"].shift(-1)
    orders["days_to_next"] = orders["next_day"] - orders["day"]
    orders["repeat30d"] = (orders["days_to_next"] <= 30).fillna(False).astype(int)
    
    # Tạo biến RFM cơ bản (Features)
    orders["prev_day"] = orders.groupby("household_key")["day"].shift(1)
    orders["recency"] = (orders["day"] - orders["prev_day"]).fillna(999)
    orders["frequency"] = orders.groupby("household_key").cumcount()
    
    print(f"   [Repeat30d] Saving repeat30d features: {orders.shape} rows")
    save_processed_data(orders, "repeat30d.parquet")