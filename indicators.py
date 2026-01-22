import pandas as pd
import numpy as np

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close_prev = df['Close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def add_indicators(df):
    # Indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'], period=14)
    df['ATR'] = calculate_atr(df, period=14)
    df['SMA_20_Slope'] = df['SMA_20'].diff()
    return df

def load_and_preprocess_data(filepath='SPY_Hourly.csv'):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None

    # Normalize column names (capitalize first letter)
    df.columns = [col.capitalize() if col.lower() in ['open', 'high', 'low', 'close', 'volume', 'date'] else col for col in df.columns]

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    df.sort_index(inplace=True)

    df = add_indicators(df)

    df.dropna(inplace=True)

    return df

def split_data(df, train_ratio=0.8):
    # Use percentage-based split instead of hardcoded dates
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df
