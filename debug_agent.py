import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from indicators import load_and_preprocess_data, split_data
from trading_environment import ForexTradingEnvironment

def debug_agent():
    # Load Data
    data_path = 'alpaca_data.csv'
    df = load_and_preprocess_data(data_path)
    train_df, test_df = split_data(df)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Load Model
    try:
        model = PPO.load("ppo_forex_agent")
    except FileNotFoundError:
        print("Model not found. Please train the agent first.")
        return

    # Debug on test data
    env = ForexTradingEnvironment(test_df)
    obs, _ = env.reset()
    
    # Track actions
    action_counts = {0: 0, 1: 0}  # Hold vs Trade
    
    print("\n=== First 20 Steps ===")
    for i in range(min(20, len(test_df) - env.window_size)):
        action, _states = model.predict(obs, deterministic=True)
        decision, direction, sl_idx, tp_idx = action
        
        action_counts[decision] += 1
        
        print(f"Step {i}: Decision={decision} (0=Hold, 1=Trade), "
              f"Direction={direction} (0=Long, 1=Short), "
              f"SL={sl_idx}, TP={tp_idx}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    print(f"\n=== Summary ===")
    print(f"Hold actions: {action_counts[0]}")
    print(f"Trade actions: {action_counts[1]}")
    print(f"Trades executed: {len(env.trades_history)}")
    print(f"Final balance: {env.balance}")
    
    # Run full episode
    print("\n=== Running Full Episode ===")
    env2 = ForexTradingEnvironment(test_df)
    obs, _ = env2.reset()
    done = False
    total_actions = {0: 0, 1: 0}
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        total_actions[action[0]] += 1
        obs, reward, terminated, truncated, info = env2.step(action)
        done = terminated or truncated
    
    print(f"Total Hold actions: {total_actions[0]}")
    print(f"Total Trade actions: {total_actions[1]}")
    print(f"Total trades completed: {len(env2.trades_history)}")
    print(f"Final balance: {env2.balance}")
    
    if env2.trades_history:
        print("\nTrade History:")
        for trade in env2.trades_history[:10]:
            print(f"  Step {trade['step']}: PnL={trade['pnl']:.6f}, Reward={trade['reward']:.2f}")

if __name__ == "__main__":
    debug_agent()
