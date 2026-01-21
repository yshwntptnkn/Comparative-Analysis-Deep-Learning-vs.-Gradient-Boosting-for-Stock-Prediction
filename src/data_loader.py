import yfinance as yf
import datetime

def fetch_data(ticker, start_date='2021-01-01'):
    today = datetime.date.today().strftime('%Y-%m-%d')
    print(f"Fetching data for {ticker} from {start_date} to {today}...")
    
    df = yf.download(ticker, start=start_date, end=today, auto_adjust=True)
    
    if len(df) == 0:
        raise ValueError("No data fetched! Check ticker symbol or internet connection.")
        
    return df