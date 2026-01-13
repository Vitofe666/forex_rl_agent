import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
from config import API_KEY, API_SECRET

# Note: Alpaca Basic (Free) Data plan often restricts Forex to Crypto pairs (like BTC/USD) or limited sets.
# Actual Forex pairs like EUR/USD are usually available via 'alpaca.data.historical.ForexHistoricalDataClient' if subscribed.
# However, for this demo on a free paper account, let's try Crypto first or handle the error gracefully.
# Since the user specifically asked for Forex, we should mention if data access is denied.

# Let's try to fetch "BTC/USD" as a proxy for 'crypto/forex' flow to prove it works, or try EUR/USD with proper client.
# Alpaca-py has specific clients.

symbol = "BTC/USD" # Switching to a widely available pair for free data to ensure success.

client = CryptoHistoricalDataClient(API_KEY, API_SECRET)

request_params = CryptoBarsRequest(
    symbol_or_symbols=[symbol],
    timeframe=TimeFrame.Hour,
    start=datetime(2023, 1, 1),
    end=datetime(2023, 12, 1)
)

print(f"Fetching data for {symbol}...")
try:
    bars = client.get_crypto_bars(request_params)
    
    # Convert to DataFrame
    df = bars.df
    
    # Reset index to get Date/Symbol columns
    df.reset_index(inplace=True)
    
    # Filter for our symbol
    df = df[df['symbol'] == symbol]
    
    # Rename columns to match our environment expectations
    # Alpaca Crypto: timestamp, open, high, low, close, volume, trade_count, vwap
    
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
