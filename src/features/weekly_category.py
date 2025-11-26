import pandas as pd
from src.io import load_processed_data, save_processed_data

def create_weekly_data():
    print("   [Weekly] Loading cleaned transactions...")
    tx = load_processed_data("tx_clean.parquet")
    
    # Gom nhóm theo Tuần + Ngành hàng (Department)
    weekly = tx.groupby(["week_no", "department"], as_index=False).agg(
        sales=("sales_value", "sum"),
        qty=("quantity", "sum"),
        avg_price=("unit_price", "mean")
    )
    
    # Đổi tên cho khớp với model
    weekly = weekly.rename(columns={"department": "group_id", "sales": "target"})
    
    # Thêm biến thời gian
    weekly["time_idx"] = weekly["week_no"] - weekly["week_no"].min()
    
    print(f"   [Weekly] Saving weekly data: {weekly.shape} rows")
    save_processed_data(weekly, "weekly_category.parquet")