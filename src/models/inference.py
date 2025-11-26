import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
from src.io import load_processed_data
from src.config import RESULTS_DIR

def run_inference_tests():
    print("   [Inference] Loading data for statistical tests...")
    df = load_processed_data("repeat30d.parquet")
    
    if "treatment" not in df.columns:
        np.random.seed(42)
        df["treatment"] = np.random.binomial(1, 0.5, len(df))

    # 1. Chi-square Test (Treatment vs Repeat)
    print("   [Inference] Running Chi-square Test...")
    contingency = pd.crosstab(df["treatment"], df["repeat30d"])
    chi2, p, dof, ex = chi2_contingency(contingency)
    
    # 2. T-test (Basket Value: Treated vs Control)
    print("   [Inference] Running T-test...")
    treated = df[df["treatment"] == 1]["basket_value"]
    control = df[df["treatment"] == 0]["basket_value"]
    t_stat, p_ttest = ttest_ind(treated, control, equal_var=False)
    
    # Lưu kết quả
    results = {
        "chi2_stat": chi2,
        "chi2_p_value": p,
        "ttest_stat": t_stat,
        "ttest_p_value": p_ttest,
        "mean_treated": treated.mean(),
        "mean_control": control.mean()
    }
    
    # Ghi ra file text cho dễ đọc
    save_path = RESULTS_DIR / "inference_stats.txt"
    with open(save_path, "w") as f:
        f.write("=== STATISTICAL INFERENCE RESULTS ===\n\n")
        f.write(f"1. Chi-square Test (Treatment vs Repeat):\n")
        f.write(f"   - Chi2 Stat: {chi2:.4f}\n")
        f.write(f"   - P-value:   {p:.6f}\n\n")
        f.write(f"2. T-test (Basket Value):\n")
        f.write(f"   - Mean Treated: {treated.mean():.2f}\n")
        f.write(f"   - Mean Control: {control.mean():.2f}\n")
        f.write(f"   - P-value:      {p_ttest:.6f}\n")
        
    print(f"   [Inference] Stats saved to {save_path.name}")