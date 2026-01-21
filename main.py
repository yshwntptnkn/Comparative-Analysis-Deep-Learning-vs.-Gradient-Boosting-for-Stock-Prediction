import argparse
import os
from tensorflow.keras.callbacks import EarlyStopping

from src.data_loader import fetch_data
from src.preprocessing import add_technical_indicators, prepare_data
from src.model_builder import build_attention_model
from src.evaluation import evaluate_model, plot_results

def main(args):
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    try:
        df = fetch_data(args.ticker)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    
    print("Preprocessing data...")
    df = add_technical_indicators(df)

    data_dict = prepare_data(df, time_window=60)
    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_val, y_val = data_dict['X_val'], data_dict['y_val']
    X_test = data_dict['X_test']

    print("Building Attention-Based LSTM Model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_attention_model(input_shape)

    print(f"Starting training for {args.epochs} epochs...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )

    model.save(f"models/{args.ticker}_attention_lstm.keras")
    print("Model saved.")

    print("Evaluating...")
    predictions = model.predict(X_test)

    scaler = data_dict['target_scaler']
    predictions_unscaled = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(data_dict['y_test'].reshape(-1, 1))

    results_df = evaluate_model(
        y_test_actual, 
        predictions_unscaled, 
        data_dict['test_dates'],
        save_path=f"results/{args.ticker}_predictions.csv"
    )

    plot_results(
        results_df, 
        title=f"{args.ticker} Price Prediction", 
        save_path=f"results/{args.ticker}_plot.png"
    )

if __name__ == "__main__":
    # This allows you to run the script with flags from the terminal
    parser = argparse.ArgumentParser(description="Stock Price Prediction Pipeline")
    parser.add_argument('--ticker', type=str, default='ITC.NS', help='Stock Ticker (e.g., ITC.NS)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    
    args = parser.parse_args()
    main(args)
    