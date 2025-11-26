import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, average_precision_score

def get_regression_metrics(y_true, y_pred):
    """Metrics cho bài toán dự báo (Forecasting/Regression)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE (xử lý chia cho 0)
    denom = np.where(y_true == 0, 1, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / denom))
    
    return {"rmse": rmse, "mae": mae, "mape": mape}

def get_classification_metrics(y_true, y_prob):
    """Metrics cho bài toán phân loại (Churn Prediction)"""
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob)
    }