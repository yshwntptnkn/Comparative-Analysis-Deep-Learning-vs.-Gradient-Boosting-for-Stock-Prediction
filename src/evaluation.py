import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import os

def evaluate_model(y_true, y_pred, dates, save_path='results/predictions.csv'):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}%")
    print(f"R^2 Score: {r2:.4f}")

    results_df = pd.DataFrame({
        'Date': dates,
        'Actual': y_true.flatten(),
        'Predicted': y_pred.flatten()
    })
    results_df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")

    return results_df

def plot_results(df, title='Stock Price Prediction', save_path='results/plot.png'):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Actual'], color='blue', label='Actual')
    plt.plot(df['Date'], df['Predicted'], color='red', label='Predicted')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    # plt.show() # Uncomment if you want to see it pop up

# Append this to src/evaluation.py

def compare_models(lstm_path, xgb_path, save_dir='results'):
    try:
        df_lstm = pd.read_csv(lstm_path)
        df_xgb = pd.read_csv(xgb_path)
    except FileNotFoundError:
        print("Error: Prediction files not found.")
        return

    df_lstm['Date'] = pd.to_datetime(df_lstm['Date'])
    df_xgb['Date'] = pd.to_datetime(df_xgb['Date'])

    # Inner join to compare only overlapping dates
    df = pd.merge(df_lstm, df_xgb, on='Date', how='inner', suffixes=('_lstm', '_xgb'))
    
    plt.figure(figsize=(14, 7))
    
    plt.plot(df['Date'], df['Actual_Price_lstm'], label="Actual Price", color='black', linewidth=2, alpha=0.8)
    
    plt.plot(df['Date'], df['LSTM_Prediction'], label="Attention LSTM", color='#e63946', linestyle='--', linewidth=1.5)
    
    plt.plot(df['Date'], df['XGBoost_Prediction'], label="XGBoost", color='#2a9d8f', linestyle='-.', linewidth=1.5)
    
    plt.title("Model Benchmark: Deep Learning vs. Gradient Boosting", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'benchmark_chart.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Benchmark chart saved to '{save_path}'")