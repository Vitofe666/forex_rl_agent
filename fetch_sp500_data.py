import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
from config import API_KEY, API_SECRET

# Fetch SPY (S&P 500 ETF) daily data
symbol = "SPY"

client = StockHistoricalDataClient(API_KEY, API_SECRET)

request_params = StockBarsRequest(
    symbol_or_symbols=[symbol],
    timeframe=TimeFrame.Day,
    start=datetime(2022, 1, 1),
    end=datetime(2024, 12, 31)
)

print(f"Fetching daily data for {symbol}...")
try:
    bars = client.get_stock_bars(request_params)
    
    # Convert to DataFrame
    df = bars.df
    
    # Reset index to get Date/Symbol columns
    df.reset_index(inplace=True)
    
    # Filter for our symbol
    df = df[df['symbol'] == symbol]
    
    # Rename columns to match our environment expectations
    df.rename(columns={
        'timestamp': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    
    # Keep only necessary columns
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Save to CSV
    output_file = 'alpaca_data.csv'
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}. Rows: {len(df)}")
    
except Exception as e:
    print(f"Error fetching data: {e}")
    print("Ensure you have access to the requested data feed.")
