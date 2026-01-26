import pandas as pd
import numpy as np
import gymnasium
import stable_baselines3
from indicators import load_and_preprocess_data
from trading_environment import ForexTradingEnvironment

try:
    print("Imports successful")
    df = load_and_preprocess_data('SPY_Hourly.csv')
    print(f"Data loaded: {len(df)}")
    env = ForexTradingEnvironment(df)
    obs, _ = env.reset()
    print(f"Env reset, obs shape: {obs.shape}")
    print("Debug Check Passed")
except Exception as e:
    print(f"Debug Failed: {e}")
    import traceback
    traceback.print_exc()
