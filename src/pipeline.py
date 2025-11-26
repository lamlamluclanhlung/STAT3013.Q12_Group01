"""
Pipeline chính: Chạy toàn bộ quy trình từ xử lý dữ liệu đến huấn luyện mô hình.
Đã loại bỏ SEM theo yêu cầu.
Lệnh chạy: python -m src.pipeline
"""

from src.utils import seed_all
from src.features import basket, repeat30d, weekly_category
from src.models import elasticity, forecasting, repeat_uplift, inference

def run_feature_engineering():
    print("\n>>> [PHASE 1] Feature Engineering (Creating Data)...")
    
    print("    -> Processing Basket Data...")
    basket.clean_transactions()
    
    print("    -> Creating Repeat30d Targets...")
    repeat30d.create_repeat_target()
    
    print("    -> Creating Weekly Category Data...")
    weekly_category.create_weekly_data()

def run_analytics_and_modeling():
    print("\n>>> [PHASE 2] Analytics & Modeling...")
    
    print("    -> Running Inference Tests (T-test/Chi-square)...")
    inference.run_inference_tests()
    
    print("    -> Running Elasticity Analysis (OLS)...")
    elasticity.run_elasticity_analysis()
    
    print("    -> Running Forecasting (ARIMA)...")
    forecasting.run_forecasting()
    
    print("    -> Running Uplift Modeling & ROI Analysis...")
    repeat_uplift.run_uplift_modeling()

def run_all():
    # Cố định random seed
    seed_all(42)
    
    run_feature_engineering()

    run_analytics_and_modeling()
    
    print("\n>>> ✅ PROJECT COMPLETED SUCCESSFULLY!")
    print(">>> Check results in 'results/' and figures in 'figures/'")

if __name__ == "__main__":
    run_all()