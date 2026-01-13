import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class ForexTradingEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size=30, initial_balance=10000.0, spread=0.0002):
        super(ForexTradingEnvironment, self).__init__()

        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.spread = spread # Spread in price (e.g., 2 pips = 0.0002)

        # Actions: 
        # 0: Trade Decision (0=Hold, 1=Enter Trade)
        # 1: Direction (0=Long, 1=Short)
        # 2: SL (0=60, 1=90, 2=120) (in pips)
        # 3: TP (0=60, 1=90, 2=120) (in pips)
        self.action_space = spaces.MultiDiscrete([2, 2, 3, 3])

        # State: Window of data
        # Features: Open, High, Low, Close, Volume, RSI, SMA_20, SMA_50, ATR, SMA_20_Slope
        # (Assuming these columns exist in df)
        self.feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'SMA_20', 'SMA_50', 'ATR', 'SMA_20_Slope']
        self.n_features = len(self.feature_cols)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size, self.n_features), 
            dtype=np.float32
        )

        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.active_trade = None # {'type': 'long'/'short', 'entry_price': float, 'sl': float, 'tp': float}
        self.trades_history = []
        
        # SL/TP as percentage of price (works for both Forex and Crypto)
        # 0=1%, 1=2%, 2=3%
        self.sl_options = [0.01, 0.02, 0.03]  # 1%, 2%, 3% of entry price
        self.tp_options = [0.01, 0.02, 0.03]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.active_trade = None
        self.trades_history = []
        return self._next_observation(), {}

    def _next_observation(self):
        # Get window
        obs = self.df.iloc[self.current_step - self.window_size : self.current_step][self.feature_cols].values
        return obs.astype(np.float32)

    def step(self, action):
        # action is [decision, direction, sl_idx, tp_idx]
        decision, direction_idx, sl_idx, tp_idx = action
        
        current_price = self.df.iloc[self.current_step]['Close']
        current_high = self.df.iloc[self.current_step]['High']
        current_low = self.df.iloc[self.current_step]['Low']
        
        reward = 0
        terminated = False
        truncated = False
        
        # Negative reward for not being in a trade (opportunity cost)
        # This encourages the agent to explore trading rather than always holding
        if not self.active_trade:
            reward = -0.1  # Increased penalty for idle holding to encourage trading
        
        # Check active trade exit
        if self.active_trade:
            # Check if SL or TP hit
            # Case: Ambiguous (Both hit in same candle) -> Loss
            # We check Low <= SL and High >= TP for Long, etc.
            
            trade = self.active_trade
            pnl = 0
            closed = False
            
            if trade['type'] == 'long':
                sl_hit = current_low <= trade['sl']
                tp_hit = current_high >= trade['tp']
                
                if sl_hit and tp_hit:
                    # Ambiguous -> Treat as Loss (SL hit)
                    pnl = trade['sl'] - trade['entry_price']
                    closed = True
                elif sl_hit:
                    pnl = trade['sl'] - trade['entry_price']
                    closed = True
                elif tp_hit:
                    pnl = trade['tp'] - trade['entry_price']
                    closed = True
                    
            elif trade['type'] == 'short':
                sl_hit = current_high >= trade['sl']
                tp_hit = current_low <= trade['tp']
                
                if sl_hit and tp_hit:
                    # Ambiguous -> Loss
                    pnl = trade['entry_price'] - trade['sl']
                    closed = True
                elif sl_hit:
                    pnl = trade['entry_price'] - trade['sl']
                    closed = True
                elif tp_hit:
                    pnl = trade['entry_price'] - trade['tp']
                    closed = True
            
            if closed:
                # Calculate Reward
                # "Final P&L... multiplied by a factor of 10,000"
                reward = pnl * 10000
                self.balance += pnl * 100000 # Example leverage/lot size scaling? Report says "Equity... $10,000". 
                # Wait, Report says: "reward is directly proportional to the realized profit and loss (P&L)... multiplied by 10,000".
                # Real P&L in currency depends on lot size. The report is vague on lot size. 
                # "a profitable trade yields a positive reward".
                # Assuming 1 Standard Lot (100,000 units) -> 1 pip ($0.0001) = $10.
                # If Price moves 0.0060 (60 pips), PnL = 0.0060.
                # Reward = 0.0060 * 10000 = 60.
                # If we assume 1 Lot, Real PnL = 0.0060 * 100,000 = $600.
                # I will track "Equity" roughly.
                # Let's assume standard PnL for balance tracking is 1 unit.
                self.trades_history.append({'step': self.current_step, 'pnl': pnl, 'reward': reward})
                self.active_trade = None
            else:
                # Provide small unrealized PnL signal while trade is open
                if trade['type'] == 'long':
                    unrealized_pnl = current_price - trade['entry_price']
                else:
                    unrealized_pnl = trade['entry_price'] - current_price
                # Small reward based on unrealized PnL (scaled down)
                reward = unrealized_pnl * 100  # Smaller scale than realized PnL
            
        # Execute New Trade if none active
        # (Report says "Action 1... initiates a new trade". Implies we can't open if one is open? 
        # Typically yes. I will assume single position.)
        if not self.active_trade and decision == 1:
            direction = 'long' if direction_idx == 0 else 'short'
            sl_pct = self.sl_options[sl_idx]
            tp_pct = self.tp_options[tp_idx]
            
            entry_price = current_price
            
            if direction == 'long':
                sl = entry_price * (1 - sl_pct)  # SL below entry
                tp = entry_price * (1 + tp_pct)  # TP above entry
            else:
                sl = entry_price * (1 + sl_pct)  # SL above entry
                tp = entry_price * (1 - tp_pct)  # TP below entry
            
            self.active_trade = {
                'type': direction,
                'entry_price': entry_price,
                'sl': sl,
                'tp': tp
            }
            # Bonus for entering a trade (encourages exploration)
            reward = 0.5
        
        # Advance Step
        self.current_step += 1
        if self.current_step >= len(self.df):
            terminated = True
        
        info = {'balance': self.balance}
        
        # Next Observation
        if not terminated:
            next_obs = self._next_observation()
        else:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32) # Placeholder
            
        return next_obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Balance: {self.balance}')
