import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
from .config import FIGURES_DIR

def plot_calibration_curve(y_true, y_prob, model_name="Model"):
    """Vẽ và lưu Calibration Curve"""
    plt.figure(figsize=(8, 8))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ece = np.abs(prob_true - prob_pred).mean()
    
    ax1.plot(prob_pred, prob_true, "s-", label=f"{model_name} (ECE={ece:.3f})")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(f'Calibration Plot - {model_name}')
    
    # Biểu đồ histogram phân phối xác suất
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax2.hist(y_prob, range=(0, 1), bins=10, histtype="step", lw=2)
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    
    plt.tight_layout()
    save_path = FIGURES_DIR / f"calibration_{model_name.lower()}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved figure: {save_path.name}")

def plot_roi_frontier(uplift_scores, y_true, revenue=50, cost=5):
    """Vẽ và lưu ROI Frontier"""
    # Tạo DataFrame để tính toán
    df = pd.DataFrame({'uplift': uplift_scores, 'y': y_true})
    df = df.sort_values("uplift", ascending=False).reset_index(drop=True)
    
    df['n_targeted'] = df.index + 1
    df['total_cost'] = df['n_targeted'] * cost
    df['incremental_revenue'] = df['uplift'].cumsum() * revenue
    df['roi'] = (df['incremental_revenue'] - df['total_cost']) / df['total_cost']
    
    # Vẽ biểu đồ 2 trục
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df['n_targeted'], df['incremental_revenue'], 'g-', label='Revenue Gain')
    ax1.set_xlabel('Number of Customers Targeted')
    ax1.set_ylabel('Revenue ($)', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    
    ax2 = ax1.twinx()
    ax2.plot(df['n_targeted'], df['roi']*100, 'b--', label='ROI')
    ax2.set_ylabel('ROI (%)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    plt.title('ROI Frontier')
    fig.tight_layout()
    
    save_path = FIGURES_DIR / "roi_frontier.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved figure: {save_path.name}")