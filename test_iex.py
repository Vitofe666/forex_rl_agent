from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from datetime import datetime, timedelta
from config import API_KEY, API_SECRET

try:
    print("Testing Data API with IEX Feed...")
    client = StockHistoricalDataClient(API_KEY, API_SECRET)
    
    end_time = datetime.now() - timedelta(minutes=20)
    start_time = end_time - timedelta(hours=24)
    
    request_params = StockBarsRequest(
        symbol_or_symbols=["SPY"],
        timeframe=TimeFrame.Hour,
        start=start_time,
        end=end_time,
        feed=DataFeed.IEX
    )
    
    bars = client.get_stock_bars(request_params)
    print(f"Success! Fetched {len(bars.df)} bars.")

except Exception as e:
    print(f"Error: {e}")
