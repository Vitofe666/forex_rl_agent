from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from datetime import datetime, timedelta
from config import API_KEY, API_SECRET

data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

def test_fetch():
    print("Testing Data Fetch...")
    symbol = "SPY"
    # Try different configurations
    try:
        end_time = datetime.now() - timedelta(minutes=20)
        start_time = end_time - timedelta(hours=50) # 50 hours
        
        print(f"Requesting {symbol} from {start_time} to {end_time}")
        
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Hour,
            start=start_time,
            end=end_time,
            feed=DataFeed.IEX
        )
        
        bars = data_client.get_stock_bars(request_params)
        print(f"Success! Got {len(bars.df)} rows.")
        print(bars.df.head())
        
    except Exception as e:
        print(f"Fetch Failed: {e}")

if __name__ == "__main__":
    test_fetch()
