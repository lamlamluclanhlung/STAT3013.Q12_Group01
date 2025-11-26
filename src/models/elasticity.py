import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from src.io import load_processed_data
from src.config import RESULTS_DIR

def run_elasticity_analysis():
    print("   [Elasticity] Loading weekly data...")
    df = load_processed_data("weekly_category.parquet")
    
    # Chuẩn bị dữ liệu
    df = df[df["qty"] > 0].copy()
    df["ln_q"] = np.log(df["qty"])
    df["ln_p"] = np.log(df["avg_price"].clip(lower=0.01))
    
    # 1. Baseline OLS
    print("   [Elasticity] Fitting OLS model...")
    model = smf.ols("ln_q ~ ln_p + discount_rate + weekofyear + C(group_id)", data=df).fit()
    
    # Lưu kết quả
    ols_path = RESULTS_DIR / "elasticity_ols_summary.txt"
    with open(ols_path, "w") as f:
        f.write(model.summary().as_text())
    print(f"   [Elasticity] OLS summary saved to {ols_path.name}")
    
    # 2. Neural Elasticity (Optional - Demo nhanh)
    # Phần này nếu bạn muốn chạy Neural Elasticity như trong Notebook 05 thì bỏ comment bên dưới
    # neural_eps = train_neural_elasticity(df)
    # print("   [Elasticity] Neural elasticity calculated.")

def train_neural_elasticity(df, epochs=20, hidden=64, lr=1e-3):
    """Hàm phụ trợ cho Neural Elasticity (giữ nguyên logic của bạn)"""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("   [Warning] Torch not found. Skipping Neural Elasticity.")
        return None

    X = df[["avg_price", "discount_rate", "weekofyear"]].copy()
    X["avg_price"] = np.log(X["avg_price"].replace(0, 1))
    X["weekofyear"] = X["weekofyear"] / 52.0
    y = np.log(df["qty"].replace(0, 1)).values.reshape(-1, 1)

    X_t = torch.tensor(X.values.astype("float32"))
    y_t = torch.tensor(y.astype("float32"))

    model = nn.Sequential(
        nn.Linear(3, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1)
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        opt.zero_grad(); loss.backward(); opt.step()

    X_t.requires_grad_(True)
    lnq_hat = model(X_t)
    grads = torch.autograd.grad(lnq_hat, X_t, grad_outputs=torch.ones_like(lnq_hat))[0]
    return grads[:, 0].detach().numpy()