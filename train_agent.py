import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from indicators import load_and_preprocess_data, split_data
from trading_environment import ForexTradingEnvironment
import os

def train_agent():
    # Load Data
    data_path = 'alpaca_data.csv'
    if not os.path.exists(data_path):
        print("Data not found, generating dummy data...")
        from generate_dummy_data import generate_dummy_data
        generate_dummy_data(data_path)

    df = load_and_preprocess_data(data_path)
    train_df, test_df = split_data(df)

    print(f"Training Data: {len(train_df)} rows")
    
    # Create Environment
    # We wrap it in a lambda to be compatible with DummyVecEnv
    env = DummyVecEnv([lambda: ForexTradingEnvironment(train_df)])

    # Initialize Model
    # "PPO... model-free... stable-baselines3"
    # Increased entropy coefficient for more exploration
    # Increased n_steps for better experience collection
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./ppo_forex_tensorboard/",
        ent_coef=0.05,  # Encourage exploration
        n_steps=4096,   # More experience before each update
        learning_rate=3e-4
    )

    # Train
    # Report: 50,000 timesteps
    total_timesteps = 200000
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    # Save Model
    model.save("ppo_forex_agent")
    print("Model saved to ppo_forex_agent.zip")

if __name__ == "__main__":
    train_agent()
