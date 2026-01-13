from alpaca.trading.client import TradingClient
from config import API_KEY, API_SECRET

try:
    print("Testing Trading API...")
    trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
    account = trading_client.get_account()
    print(f"Account Status: {account.status}")
    print(f"Buying Power: {account.buying_power}")
except Exception as e:
    print(f"Error: {e}")
