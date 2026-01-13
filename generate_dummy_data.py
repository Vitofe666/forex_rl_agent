import pandas as pd
import numpy as np
import datetime

def generate_dummy_data(filepath='EURUSD_Hourly.csv'):
    # Generate dates from 2020 to 2025
    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2025, 12, 31)
    # Hourly data
    dates = pd.date_range(start=start_date, end=end_date, freq='h')
    
    n_samples = len(dates)
    
    # Random walk for price
    # Start at 1.1000
    price = 1.1000
    prices = []
    
    # Simple simulation: Geometric Brownian Motion-ish
    # Volatility per hour
    volatility = 0.0005 
    
    current_price = price
    
    open_data = []
    high_data = []
    low_data = []
    close_data = []
    volume_data = []
    
    np.random.seed(42)
    
    for _ in range(n_samples):
        # Drift slightly random
        drift = np.random.normal(0, volatility)
        
        # Open is close of previous (approx)
        pct_change = drift
        
        # Calculate OHLC
        # Random bar movement
        bar_move = np.random.normal(0, volatility * 0.5)
        
        open_p = current_price
        close_p = current_price * (1 + pct_change)
        
        # High and Low
        high_p = max(open_p, close_p) * (1 + abs(np.random.normal(0, volatility * 0.2)))
        low_p = min(open_p, close_p) * (1 - abs(np.random.normal(0, volatility * 0.2)))
        
        open_data.append(open_p)
        high_data.append(high_p)
        low_data.append(low_p)
        close_data.append(close_p)
        volume_data.append(np.random.randint(100, 10000))
        
        current_price = close_p
        
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_data,
        'High': high_data,
        'Low': low_data,
        'Close': close_data,
        'Volume': volume_data
    })
    
    # Save
    df.to_csv(filepath, index=False)
    print(f"Generated dummy data at {filepath} with {n_samples} rows.")

if __name__ == "__main__":
    generate_dummy_data()
