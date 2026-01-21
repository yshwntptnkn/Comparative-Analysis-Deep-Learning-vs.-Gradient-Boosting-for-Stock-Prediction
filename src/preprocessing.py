import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def add_techinical_indicators(df):
    df = df.copy()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain/loss

    df['RSI'] = 100 - (100/(1 + rs))
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Log_Volume'] = np.log(df['Volume'] + 1)
    df.dropna(inplace=True)
    return df

def create_sequences(dataset, target_col_idx, time_window=60):
    X, y = [], []
    for i in range(time_window, len(dataset)):
        X.append(dataset[i-time_window:i]) 
        y.append(dataset[i, target_col_idx]) 
    return np.array(X), np.array(y)

def prepare_data(df, time_window=60):
    feature_columns = ['Close', 'RSI', 'EMA_50', 'Log_Volume']
    target_column = 'Close'

    n = len(df)
    train_split = int(n * 0.7)
    val_split = int(n * 0.85)

    train_df = df.iloc[:train_split]
    val_df = df.iloc[train_split:val_split]
    test_df = df.iloc[val_split:]

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(train_df[feature_columns])
    
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(train_df[[target_column]])

    train_scaled = feature_scaler.transform(train_df[feature_columns])
    val_scaled = feature_scaler.transform(val_df[feature_columns])
    test_scaled = feature_scaler.transform(test_df[feature_columns])

    val_input = np.concatenate((train_scaled[-time_window:], val_scaled))
    test_input = np.concatenate((val_scaled[-time_window:], test_scaled))

    close_column_index = feature_columns.index('Close')

    X_train, y_train = create_sequences(train_scaled, close_column_index, time_window)
    X_val, y_val = create_sequences(val_input, close_column_index, time_window)
    X_test, y_test = create_sequences(test_input, close_column_index, time_window)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'target_scaler': target_scaler,
        'test_dates': test_df.index
    }
