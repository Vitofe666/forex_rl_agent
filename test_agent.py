import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from indicators import load_and_preprocess_data, split_data
from trading_environment import ForexTradingEnvironment

def evaluate_agent(data_df, model, title="Equity Curve"):
    env = ForexTradingEnvironment(data_df)
    obs, _ = env.reset()
    done = False
    
    equity_curve = [env.balance]
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        equity_curve.append(env.balance)
        
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve)
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Equity")
    plt.grid(True)
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename)
    print(f"Saved plot to {filename}. Final Balance: {env.balance:.2f}")

def test_agent():
    # Load Data
    data_path = 'SPY_Hourly.csv'
    df = load_and_preprocess_data(data_path)
    train_df, test_df = split_data(df)
    
    # Load Model
    try:
        model = PPO.load("ppo_forex_agent")
    except FileNotFoundError:
        print("Model not found. Please train the agent first.")
        return

    print("Evaluating on Training Data (In-Sample)...")
    evaluate_agent(train_df, model, title="In-Sample Performance")
    
    print("Evaluating on Testing Data (Out-of-Sample)...")
    evaluate_agent(test_df, model, title="Out-of-Sample Performance")

if __name__ == "__main__":
    test_agent()
