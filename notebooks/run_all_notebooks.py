import os
import time
import subprocess
import sys

NOTEBOOKS = [
    "00_prep_features.ipynb",
    "01_eda_and_cleaning.ipynb", 
    "02_build_features_repeat30d.ipynb",
    "03_build_weekly_category.ipynb",
    "04_inference_tests.ipynb",
    "05_elasticity_models.ipynb",
    "06_forecasting_baseline_and_tft.ipynb",
    "07_repeat_prediction_and_uplift.ipynb"
]

def run_notebook(notebook_name):
    print(f"\n{'='*60}")
    print(f"▶️  Đang chạy: {notebook_name}...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # --- SỬA LỖI TẠI ĐÂY ---
    # Dùng sys.executable để gọi chính xác Python đang chạy
    # Thay vì gọi "jupyter", ta gọi "python -m jupyter"
    python_exe = sys.executable
    cmd = f"\"{python_exe}\" -m jupyter nbconvert --to notebook --execute --inplace \"{notebook_name}\""
    
    try:
        subprocess.check_call(cmd, shell=True)
        elapsed = time.time() - start_time
        print(f"✅ Hoàn thành: {notebook_name} trong {elapsed:.2f} giây.")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ LỖI: Không thể chạy {notebook_name}.")
        return False

def main():
    print(f"🚀 BẮT ĐẦU PIPELINE ({len(NOTEBOOKS)} notebooks)")
    print(f"   Using Python: {sys.executable}")
    
    # Kiểm tra xem đã cài nbconvert chưa
    try:
        subprocess.check_call([sys.executable, "-m", "jupyter", "nbconvert", "--version"], 
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        print("⚠️ Cảnh báo: Chưa cài thư viện nbconvert. Đang tự động cài...")
        os.system(f"\"{sys.executable}\" -m pip install nbconvert")

    total_start = time.time()
    
    for nb in NOTEBOOKS:
        if not os.path.exists(nb):
            print(f"⚠️ Cảnh báo: Không tìm thấy file {nb}, đang bỏ qua...")
            continue
            
        success = run_notebook(nb)
        if not success:
            print("\n⛔ Pipeline dừng lại do có lỗi!")
            # sys.exit(1) # Comment dòng này nếu muốn chạy tiếp dù lỗi
            
    total_time = time.time() - total_start
    print(f"\n🎉🎉🎉 TẤT CẢ ĐÃ HOÀN THÀNH! Tổng thời gian: {total_time/60:.2f} phút.")

if __name__ == "__main__":
    main()