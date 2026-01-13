import os
from alpaca.trading.client import TradingClient
from alpaca.common.exceptions import APIError
from config import API_KEY, API_SECRET, BASE_URL

def check_credentials():
    print(f"Checking credentials...")
    print(f"API Key: {API_KEY[:4]}... (Length: {len(API_KEY)})")
    print(f"Base URL: {BASE_URL}")
    
    try:
        trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
        account = trading_client.get_account()
        print("\nSUCCESS: Connection Established!")
        print(f"Account Status: {account.status}")
        print(f"Buying Power: ${account.buying_power}")
        print("\nYou are authorized.")
        
    except APIError as e:
        print(f"\nERROR: API Error - {e}")
        print("This usually means your API Key or Secret is incorrect.")
        print("Please check your APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables.")
        print("Or update config.py with valid keys.")
    except Exception as e:
        print(f"\nERROR: Unexpected Error - {e}")

if __name__ == "__main__":
    check_credentials()
