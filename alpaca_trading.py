import time
import pandas as pd
import numpy as np
import pytz
from stable_baselines3 import PPO
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.data.enums import DataFeed
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from config import API_KEY, API_SECRET, BASE_URL
from indicators import add_indicators

# Configuration
SYMBOL = "SPY"
MODEL_PATH = "ppo_forex_agent.zip"
WINDOW_SIZE = 30
QTY = 10

# Initialize Clients
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

def get_latest_data(symbol, lookback_hours=250):
    """Fetches enough historical data to calculate indicators."""
    end_time = datetime.now() - timedelta(minutes=20)
    start_time = end_time - timedelta(hours=lookback_hours)
    
    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Hour,
        start=start_time,
        end=end_time,
        feed=DataFeed.IEX
    )
    
    bars = data_client.get_stock_bars(request_params)
    df = bars.df
    df.reset_index(inplace=True)
    df = df[df['symbol'] == symbol]
    
    # Rename and Clean
    df.rename(columns={
        'timestamp': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    
    # Calculate Indicators
    df = add_indicators(df)
    
    # Drop rows with NaN (critical for indicators like RSI which need warmup)
    df = df.dropna()
    
    return df

def execute_trade(action, current_price):
    """Executes trade based on agent action."""
    decision, direction_idx, sl_idx, tp_idx = action
    
    if decision == 0:
        print("Agent Decision: HOLD")
        return

    # Map Actions
    direction = OrderSide.BUY if direction_idx == 0 else OrderSide.SELL
    
    # SL/TP Options (from environment - percentages)
    sl_options = [0.01, 0.02, 0.03]
    tp_options = [0.01, 0.02, 0.03]
    
    sl_pct = sl_options[sl_idx]
    tp_pct = tp_options[tp_idx]
    
    if direction == OrderSide.BUY:
        sl_price = current_price * (1 - sl_pct)
        tp_price = current_price * (1 + tp_pct)
    else:
        sl_price = current_price * (1 + sl_pct)
        tp_price = current_price * (1 - tp_pct)
        
    print(f"Agent Decision: ENTER {direction} @ {current_price}")
    print(f"SL: {sl_price:.5f}, TP: {tp_price:.5f}")
    
    order_data = MarketOrderRequest(
        symbol=SYMBOL,
        qty=QTY,
        side=direction,
        time_in_force=TimeInForce.DAY,
        take_profit=TakeProfitRequest(limit_price=round(tp_price, 5)),
        stop_loss=StopLossRequest(stop_price=round(sl_price, 5))
    )
    
    try:
        order = trading_client.submit_order(order_data)
        print(f"Order Submitted: {order.id}")
    except Exception as e:
        print(f"Order Failed: {e}")

def main():
    print("Loading Model...")
    model = PPO.load(MODEL_PATH)
    print("Model Loaded. Starting Continuous Trading Loop (Hourly Checks)...")
    
    while True:
        try:
            print(f"\n--- Checking Market at {datetime.now()} ---")
            
            print("Checking Open Positions...")
            try:
                positions = trading_client.get_all_positions()
                has_position = any(p.symbol == SYMBOL for p in positions)
                
                if has_position:
                    print(f"Position already open for {SYMBOL}. Agent waits for exit (SL/TP).")
                    time.sleep(3600)
                    continue
                    
            except Exception as e:
                print(f"Error checking positions: {e}")
                time.sleep(60)
                continue

            print("Fetching Market Data...")
            try:
                df = get_latest_data(SYMBOL)
                print(f"Fetched {len(df)} rows after cleaning.")
                
                # Check if we have enough data
                if len(df) < WINDOW_SIZE:
                    print(f"Not enough data for window. Need {WINDOW_SIZE}, got {len(df)}")
                    time.sleep(3600)
                    continue
                
                # Prepare Observation - FIXED
                feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'SMA_20', 'SMA_50', 'ATR', 'SMA_20_Slope']
                
                # Get last WINDOW_SIZE rows for proper observation shape
                obs = df[feature_cols].iloc[-WINDOW_SIZE:].values.astype(np.float32)
                
                # Final safety check for NaN/Inf values
                if np.isnan(obs).any() or np.isinf(obs).any():
                    print("Warning: Observation contains NaN or Inf values. Skipping this iteration.")
                    print(f"NaN count: {np.isnan(obs).sum()}, Inf count: {np.isinf(obs).sum()}")
                    time.sleep(3600)
                    continue
                
                # Verify correct shape
                print(f"Observation shape: {obs.shape}")  # Should be (30, 10)
                
                # Predict
                action, _states = model.predict(obs, deterministic=True)
                
                # Get Current Price for Order logic
                current_price = df.iloc[-1]['Close']
                
                execute_trade(action, current_price)
                
            except Exception as e:
                print(f"Error in trading loop: {e}")
                import traceback
                traceback.print_exc()
                
            print("Sleeping for 1 hour...")
            time.sleep(3600)
            
        except KeyboardInterrupt:
            print("Trading Loop Stopped by User.")
            break
        except Exception as e:
            print(f"Critical Error in Main Loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(60)

if __name__ == '__main__':
    main()
