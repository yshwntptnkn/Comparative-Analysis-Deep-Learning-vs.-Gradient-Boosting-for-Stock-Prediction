# benchmark.py
from src.evaluation import compare_models

if __name__ == "__main__":
    print("Running Model Benchmark...")
    
    # Call the function we just added to src/evaluation.py
    compare_models(
        lstm_path='results/ITC.NS_predictions.csv',     # Output from main.py
        xgb_path='results/ITC.NS_xgb_predictions.csv',  # Output from train_xgboost.py
        save_dir='results'
    )
    
    print("Benchmarking Complete. Check 'results/' folder.")