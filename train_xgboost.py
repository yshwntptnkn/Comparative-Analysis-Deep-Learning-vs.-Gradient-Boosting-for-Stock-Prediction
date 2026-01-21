import argparse
import xgboost as xgb
import pandas as pd
import numpy as np
import os
from src.data_loader import fetch_data
from src.evaluation import evaluate_model, plot_results

def prepare_xgb_data(df):
    df = df.copy()
    
    # Feature Engineering
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Returns & Lags
    df['Daily_Return'] = df['Close'].pct_change()
    df['Log_Volume'] = np.log(df['Volume'] + 1)
    
    # Create Lags 
    for lag in [1, 2, 3, 5, 10]:
        df[f'Return_Lag_{lag}'] = df['Daily_Return'].shift(lag)
        df[f'RSI_Lag_{lag}'] = df['RSI'].shift(lag)
        
    # Target: Predict NEXT day's return
    df['Target'] = df['Daily_Return'].shift(-1)
    
    df.dropna(inplace=True)
    return df

def main(args):
    os.makedirs('results', exist_ok=True)
    
    # 1. Fetch Data (Reusing your src module)
    df = fetch_data(args.ticker)
    
    # 2. Prepare Data
    print("Preparing data for XGBoost...")
    df = prepare_xgb_data(df)
    
    # Split
    train_size = int(len(df) * 0.85)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Define features (Exclude target and raw prices)
    features = [col for col in df.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
    X_train, y_train = train_df[features], train_df['Target']
    X_test, y_test = test_df[features], test_df['Target']
    
    # 3. Train
    print("Training XGBoost...")
    reg = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        early_stopping_rounds=50,
        random_state=42
    )
    
    reg.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100
    )
    
    # 4. Predict & Reconstruct Prices
    pred_returns = reg.predict(X_test)
    
    previous_closes = test_df['Close'].values[:-1] # Drop last row (no target)
    actual_prices = test_df['Close'].shift(-1).dropna().values
    pred_returns = pred_returns[:-1]
    test_dates = test_df.index[:-1]
    
    predicted_prices = previous_closes * (1 + pred_returns)
    
    # 5. Evaluate
    results_df = evaluate_model(
        actual_prices,
        predicted_prices,
        test_dates,
        save_path=f"results/{args.ticker}_xgb_predictions.csv"
    )
    
    plot_results(
        results_df,
        title=f"{args.ticker} - XGBoost Prediction",
        save_path=f"results/{args.ticker}_xgb_plot.png"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='ITC.NS')
    main(parser.parse_args())