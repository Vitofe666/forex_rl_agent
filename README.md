# Forex RL Agent

This project implements a Reinforcement Learning (RL) agent for Forex trading using `stable-baselines3`.

## Prerequisites

Ensure you have Python installed (3.8+ recommended).

## Installation

1.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Training the Agent

To train the agent, run the `train_agent.py` script. This will:
-   Generate dummy data (`EURUSD_Hourly.csv`) if it doesn't exist.
-   Train a PPO model for 50,000 timesteps.
-   Save the trained model to `ppo_forex_agent.zip`.

```bash
python train_agent.py
```

### 2. Testing the Agent

After training, you can evaluate the agent's performance on both in-sample (training) and out-of-sample (testing) data.

```bash
python test_agent.py
```

This will generate performance plots (e.g., `equity_curve.png`) and print the final balance.

## Files

-   `train_agent.py`: Main script to train the model.
-   `test_agent.py`: Script to evaluate the trained model.
-   `trading_environment.py`: Custom Gym environment for Forex trading.
-   `indicators.py`: Helper functions for calculating technical indicators.
-   `generate_dummy_data.py`: Generates synthetic market data for testing.
