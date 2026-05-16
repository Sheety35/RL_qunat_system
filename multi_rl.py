import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque
import random
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None

class PortfolioTradingEnvironment(gym.Env):
    """Multi-Stock Portfolio Trading Environment for RL Training"""
    
    def __init__(self, stock_data_dict, initial_balance=1000, transaction_cost=0.001, 
                 max_loss_pct=0.30, profit_target_pct=0.50, max_position_pct=0.25):
        super(PortfolioTradingEnvironment, self).__init__()
        
        self.stock_data_dict = stock_data_dict
        self.symbols = list(stock_data_dict.keys())
        self.num_stocks = len(self.symbols)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_loss_pct = max_loss_pct
        self.profit_target_pct = profit_target_pct
        self.max_position_pct = max_position_pct  # Max 25% in any single stock
        
        # Risk thresholds
        self.stop_loss_threshold = initial_balance * (1 - max_loss_pct)
        self.profit_target = initial_balance * (1 + profit_target_pct)
        
        # Ensure all stocks have same length
        min_length = min(len(data) for data in stock_data_dict.values())
        for symbol in self.symbols:
            self.stock_data_dict[symbol] = self.stock_data_dict[symbol].iloc[:min_length].reset_index(drop=True)
        
        # Action space: For each stock [Hold=0, Buy=1, Sell=2, Buy_More=3, Sell_Half=4]
        # Total actions = num_stocks * 5
        self.action_space = spaces.Discrete(self.num_stocks * 5)
        
        # Observation space: Portfolio state + each stock's features
        # Portfolio: balance_ratio, total_portfolio_value, diversification_ratio, risk_exposure
        # Per stock: price, volume, sma5, sma20, rsi, holdings, position_value, price_change
        obs_size = 4 + (self.num_stocks * 8)  # 4 portfolio + 8 per stock
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_size,), dtype=np.float32
        )
        
        print(f"📊 Portfolio Environment Created:")
        print(f"   • Stocks: {self.symbols}")
        print(f"   • Data length: {min_length} days")
        print(f"   • Action space: {self.action_space.n} actions")
        print(f"   • Observation space: {obs_size} features")
        print(f"   • Max position per stock: {max_position_pct*100}%")
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = {symbol: 0 for symbol in self.symbols}
        self.total_trades = 0
        self.trades_per_stock = {symbol: 0 for symbol in self.symbols}
        
        return self._get_observation()
    
    def _get_current_prices(self):
        """Get current prices for all stocks"""
        if self.current_step >= len(list(self.stock_data_dict.values())[0]):
            # Return last known prices
            last_step = len(list(self.stock_data_dict.values())[0]) - 1
            return {symbol: float(data.iloc[last_step]['Close']) 
                   for symbol, data in self.stock_data_dict.items()}
        
        return {symbol: float(data.iloc[self.current_step]['Close']) 
                for symbol, data in self.stock_data_dict.items()}
    
    def _calculate_portfolio_value(self):
        """Calculate total portfolio value"""
        current_prices = self._get_current_prices()
        stock_values = sum(self.shares_held[symbol] * current_prices[symbol] 
                          for symbol in self.symbols)
        return self.balance + stock_values
    
    def _get_stock_features(self, symbol):
        """Get technical features for a specific stock"""
        if self.current_step >= len(self.stock_data_dict[symbol]):
            return np.zeros(8)
        
        data = self.stock_data_dict[symbol]
        current_price = float(data.iloc[self.current_step]['Close'])
        
        # Technical indicators
        sma_5 = self._get_sma(symbol, 5)
        sma_20 = self._get_sma(symbol, 20)
        rsi = self._get_rsi(symbol)
        
        # Position info
        current_holdings = self.shares_held[symbol]
        position_value = current_holdings * current_price
        
        # Price change
        price_change = 0
        if self.current_step > 0:
            prev_price = float(data.iloc[self.current_step - 1]['Close'])
            price_change = (current_price - prev_price) / prev_price if prev_price > 0 else 0
        
        # Volume (normalized)
        volume = float(data.iloc[self.current_step]['Volume']) / 1000000
        
        return np.array([
            current_price / 100,  # Normalized price
            volume,
            sma_5 / 100,
            sma_20 / 100,
            rsi / 100,
            current_holdings / 100,  # Normalized holdings
            position_value / self.initial_balance,  # Position as % of initial balance
            price_change
        ])
    
    def _get_observation(self):
        """Get current state observation for entire portfolio"""
        if self.current_step >= len(list(self.stock_data_dict.values())[0]):
            return np.zeros(self.observation_space.shape[0])
        
        # Portfolio-level features
        portfolio_value = self._calculate_portfolio_value()
        balance_ratio = self.balance / self.initial_balance
        portfolio_return = (portfolio_value - self.initial_balance) / self.initial_balance
        
        # Diversification ratio (how spread out investments are)
        current_prices = self._get_current_prices()
        position_values = [self.shares_held[symbol] * current_prices[symbol] for symbol in self.symbols]
        total_invested = sum(position_values)
        
        if total_invested > 0:
            # Calculate Herfindahl index (concentration measure)
            weights = [val / total_invested for val in position_values]
            diversification_ratio = 1 - sum(w**2 for w in weights)  # 1 = perfectly diversified, 0 = concentrated
        else:
            diversification_ratio = 0
        
        # Risk exposure (total invested / portfolio value)
        risk_exposure = total_invested / portfolio_value if portfolio_value > 0 else 0
        
        portfolio_features = np.array([
            balance_ratio,
            portfolio_return,
            diversification_ratio,
            risk_exposure
        ])
        
        # Individual stock features
        stock_features = []
        for symbol in self.symbols:
            stock_features.extend(self._get_stock_features(symbol))
        
        # Combine all features
        obs = np.concatenate([portfolio_features, stock_features])
        return obs.astype(np.float32)
    
    def _get_sma(self, symbol, window):
        """Simple Moving Average for a specific stock"""
        if self.current_step < window:
            return float(self.stock_data_dict[symbol].iloc[0]['Close'])
        
        start_idx = max(0, self.current_step - window + 1)
        sma_value = self.stock_data_dict[symbol].iloc[start_idx:self.current_step + 1]['Close'].mean()
        return float(sma_value)
    
    def _get_rsi(self, symbol, window=14):
        """Relative Strength Index for a specific stock"""
        if self.current_step < window:
            return 50.0
        
        start_idx = max(0, self.current_step - window)
        prices = self.stock_data_dict[symbol].iloc[start_idx:self.current_step + 1]['Close']
        
        delta = prices.diff().dropna()
        if len(delta) == 0:
            return 50.0
            
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gain = float(gains.mean())
        avg_loss = float(losses.mean())
        
        if avg_loss == 0 or pd.isna(avg_loss) or np.isnan(avg_loss):
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        if pd.isna(rsi) or np.isnan(rsi):
            return 50.0
        
        return float(rsi)
    
    def step(self, action):
        if self.current_step >= len(list(self.stock_data_dict.values())[0]) - 1:
            return self._get_observation(), 0, True, self._get_info()
        
        # Decode action: which stock and what action
        stock_idx = action // 5
        action_type = action % 5
        symbol = self.symbols[stock_idx]
        
        current_prices = self._get_current_prices()
        current_price = current_prices[symbol]
        portfolio_value = self._calculate_portfolio_value()
        
        # Execute action
        reward = 0
        executed = False
        
        # Check position limits
        current_position_value = self.shares_held[symbol] * current_price
        current_position_pct = current_position_value / portfolio_value if portfolio_value > 0 else 0
        
        if action_type == 1:  # Buy
            max_investment = min(
                self.balance * 0.8,  # Don't invest all cash
                portfolio_value * self.max_position_pct - current_position_value  # Position limit
            )
            
            if max_investment > current_price * (1 + self.transaction_cost):
                shares_to_buy = int(max_investment / (current_price * (1 + self.transaction_cost)))
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                    self.balance -= cost
                    self.shares_held[symbol] += shares_to_buy
                    self.total_trades += 1
                    self.trades_per_stock[symbol] += 1
                    executed = True
                    reward += 0.5  # Small reward for taking action
        
        elif action_type == 2 and self.shares_held[symbol] > 0:  # Sell all
            sell_value = self.shares_held[symbol] * current_price * (1 - self.transaction_cost)
            self.balance += sell_value
            self.shares_held[symbol] = 0
            self.total_trades += 1
            self.trades_per_stock[symbol] += 1
            executed = True
            
        elif action_type == 3:  # Buy more (smaller amount)
            max_investment = min(
                self.balance * 0.4,  # Smaller buy
                portfolio_value * self.max_position_pct - current_position_value
            )
            
            if max_investment > current_price * (1 + self.transaction_cost):
                shares_to_buy = int(max_investment / (current_price * (1 + self.transaction_cost)))
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                    self.balance -= cost
                    self.shares_held[symbol] += shares_to_buy
                    self.total_trades += 1
                    self.trades_per_stock[symbol] += 1
                    executed = True
                    reward += 0.25
        
        elif action_type == 4 and self.shares_held[symbol] > 1:  # Sell half
            shares_to_sell = self.shares_held[symbol] // 2
            sell_value = shares_to_sell * current_price * (1 - self.transaction_cost)
            self.balance += sell_value
            self.shares_held[symbol] -= shares_to_sell
            self.total_trades += 1
            self.trades_per_stock[symbol] += 1
            executed = True
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward based on portfolio performance
        if self.current_step < len(list(self.stock_data_dict.values())[0]):
            new_portfolio_value = self._calculate_portfolio_value()
            
            # Portfolio return reward
            portfolio_return = (new_portfolio_value - self.initial_balance) / self.initial_balance
            reward += portfolio_return * 100
            
            # Diversification bonus
            current_prices = self._get_current_prices()
            position_values = [self.shares_held[s] * current_prices[s] for s in self.symbols]
            total_invested = sum(position_values)
            
            if total_invested > 0:
                weights = [val / total_invested for val in position_values if val > 0]
                if len(weights) > 1:  # Bonus for diversification
                    diversification_score = len(weights) / self.num_stocks
                    reward += diversification_score * 2
            
            # Risk management penalties/rewards
            if new_portfolio_value <= self.stop_loss_threshold:
                reward -= 100  # Heavy penalty for large losses
            elif new_portfolio_value >= self.profit_target:
                reward += 50   # Bonus for reaching profit target
            
            # Overtrading penalty
            if self.total_trades > 100:
                reward -= 0.1 * (self.total_trades - 100)
            
            # Position concentration penalty
            for symbol in self.symbols:
                position_value = self.shares_held[symbol] * current_prices[symbol]
                position_pct = position_value / new_portfolio_value if new_portfolio_value > 0 else 0
                if position_pct > self.max_position_pct:
                    reward -= (position_pct - self.max_position_pct) * 20  # Penalty for concentration
        
        done = (self.current_step >= len(list(self.stock_data_dict.values())[0]) - 1 or 
                new_portfolio_value <= self.stop_loss_threshold or 
                new_portfolio_value >= self.profit_target)
        
        return self._get_observation(), reward, done, self._get_info()
    
    def _get_info(self):
        """Get detailed info about current state"""
        portfolio_value = self._calculate_portfolio_value()
        current_prices = self._get_current_prices()
        
        positions = {}
        for symbol in self.symbols:
            position_value = self.shares_held[symbol] * current_prices[symbol]
            positions[symbol] = {
                'shares': self.shares_held[symbol],
                'value': position_value,
                'percentage': position_value / portfolio_value * 100 if portfolio_value > 0 else 0,
                'trades': self.trades_per_stock[symbol]
            }
        
        return {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'total_trades': self.total_trades,
            'return_pct': (portfolio_value - self.initial_balance) / self.initial_balance * 100,
            'positions': positions,
            'step': self.current_step
        }

class PortfolioDQNAgent:
    """Enhanced DQN Agent for Portfolio Trading"""
    
    def __init__(self, state_size, action_size, learning_rate=0.0005):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)  # Larger memory for complex decisions
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.gamma = 0.95
        
        # Build networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
        
        print(f"🤖 Portfolio Agent Created:")
        print(f"   • State size: {state_size}")
        print(f"   • Action size: {action_size}")
        print(f"   • Memory capacity: {self.memory.maxlen}")
    
    def _build_model(self):
        """Build enhanced neural network for portfolio decisions"""
        model = models.Sequential([
            layers.Dense(256, input_dim=self.state_size, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='huber'  # More robust to outliers than MSE
        )
        return model
    
    def update_target_network(self):
        """Copy weights to target network"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Epsilon-greedy action selection with action masking"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size=128):
        """Train the model with larger batch size"""
        if len(self.memory) < batch_size:
            return 0
        
        batch = random.sample(self.memory, batch_size)
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        
        current_q_values = self.q_network.predict(states, verbose=0)
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        targets = current_q_values.copy()
        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_q_values[i])
        
        history = self.q_network.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss

class PortfolioTradingTrainer:
    """Portfolio Trading Model Trainer"""
    
    def __init__(self, symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA'], 
                 start_date='2019-01-01', end_date='2024-01-01',
                 initial_balance=10000, max_loss_pct=0.30, profit_target_pct=0.50):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.max_loss_pct = max_loss_pct
        self.profit_target_pct = profit_target_pct
        
        os.makedirs("portfolio_models", exist_ok=True)
        
        print(f"🎯 Portfolio Trading Configuration:")
        print(f"   • Stocks: {symbols}")
        print(f"   • Initial Balance: ${initial_balance}")
        print(f"   • Max Loss: {max_loss_pct*100}%")
        print(f"   • Profit Target: {profit_target_pct*100}%")
    
    def fetch_portfolio_data(self):
        """Fetch data for all stocks in portfolio"""
        print(f"\n📊 Fetching portfolio data...")
        
        stock_data_dict = {}
        for symbol in self.symbols:
            try:
                data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
                if len(data) > 0:
                    stock_data_dict[symbol] = data.dropna()
                    print(f"   ✅ {symbol}: {len(data)} days")
                else:
                    print(f"   ❌ {symbol}: No data")
            except Exception as e:
                print(f"   ❌ {symbol}: {e}")
        
        if len(stock_data_dict) < 2:
            raise Exception("Need at least 2 stocks for portfolio trading")
        
        # Ensure all stocks have same date range
        min_length = min(len(data) for data in stock_data_dict.values())
        for symbol in stock_data_dict.keys():
            stock_data_dict[symbol] = stock_data_dict[symbol].iloc[:min_length]
        
        # Split data
        split_point = int(min_length * 0.85)
        train_data = {symbol: data.iloc[:split_point] for symbol, data in stock_data_dict.items()}
        val_data = {symbol: data.iloc[split_point:] for symbol, data in stock_data_dict.items()}
        
        print(f"   📈 Portfolio data: {min_length} days")
        print(f"   🏋️ Training: {split_point} days")
        print(f"   ✅ Validation: {min_length - split_point} days")
        
        return train_data, val_data
    
    def train_portfolio_agent(self, episodes=2000, save_frequency=200):
        """Train the portfolio agent"""
        print(f"\n🚀 Starting portfolio training for {episodes} episodes...")
        
        train_data, val_data = self.fetch_portfolio_data()
        
        # Create environment and agent
        env = PortfolioTradingEnvironment(
            train_data, self.initial_balance, 
            max_loss_pct=self.max_loss_pct,
            profit_target_pct=self.profit_target_pct
        )
        
        agent = PortfolioDQNAgent(
            env.observation_space.shape[0],
            env.action_space.n
        )
        
        # Training tracking
        episode_rewards = []
        portfolio_returns = []
        validation_scores = []
        losses = []
        best_validation = float('-inf')
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    episode_rewards.append(total_reward)
                    portfolio_returns.append(info['return_pct'])
                    break
            
            # Train agent
            if len(agent.memory) > 256:
                loss = agent.replay(batch_size=128)
                losses.append(loss)
            
            # Update target network
            if episode % 100 == 0:
                agent.update_target_network()
            
            # Validation and saving
            if episode % save_frequency == 0 and episode > 0:
                val_score = self._validate_portfolio_agent(agent, val_data)
                validation_scores.append(val_score)
                
                if val_score > best_validation:
                    best_validation = val_score
                    self._save_portfolio_model(agent, episode, val_score, is_best=True)
                
                # Progress report
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                avg_return = np.mean(portfolio_returns[-100:]) if len(portfolio_returns) >= 100 else np.mean(portfolio_returns)
                avg_loss = np.mean(losses[-10:]) if losses else 0
                
                print(f"Episode {episode:4d} | "
                      f"Reward: {avg_reward:8.2f} | "
                      f"Return: {avg_return:6.2f}% | "
                      f"Val: {val_score:6.2f}% | "
                      f"Loss: {avg_loss:.4f} | "
                      f"ε: {agent.epsilon:.3f} | "
                      f"Trades: {info['total_trades']:3d}")
        
        # Final save
        final_val = self._validate_portfolio_agent(agent, val_data)
        self._save_portfolio_model(agent, episodes, final_val, is_final=True)
        
        print(f"\n✅ Portfolio training completed!")
        print(f"   🏆 Best validation: {best_validation:.2f}%")
        print(f"   📊 Final validation: {final_val:.2f}%")
        
        self._plot_portfolio_progress(episode_rewards, portfolio_returns, validation_scores)
        
        return agent, {
            'rewards': episode_rewards,
            'returns': portfolio_returns,
            'validation': validation_scores,
            'symbols': self.symbols
        }
    
    def _validate_portfolio_agent(self, agent, val_data):
        """Validate portfolio agent"""
        val_env = PortfolioTradingEnvironment(
            val_data, self.initial_balance,
            max_loss_pct=self.max_loss_pct,
            profit_target_pct=self.profit_target_pct
        )
        
        state = val_env.reset()
        original_epsilon = agent.epsilon
        agent.epsilon = 0  # No exploration during validation
        
        while True:
            action = agent.act(state)
            state, reward, done, info = val_env.step(action)
            if done:
                validation_return = info['return_pct']
                break
        
        agent.epsilon = original_epsilon
        return validation_return
    
    def _save_portfolio_model(self, agent, episode, val_score, is_best=False, is_final=False):
        """Save portfolio model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_best:
            filename = f"best_portfolio_model_{timestamp}.h5"
        elif is_final:
            filename = f"final_portfolio_model_{timestamp}.h5"
        else:
            filename = f"portfolio_model_ep{episode}_{timestamp}.h5"
        
        model_path = os.path.join("portfolio_models", filename)
        agent.q_network.save(model_path)
        
        # Save metadata
        metadata = {
            'episode': episode,
            'validation_score': val_score,
            'symbols': self.symbols,
            'initial_balance': self.initial_balance,
            'max_loss_pct': self.max_loss_pct,
            'profit_target_pct': self.profit_target_pct,
            'state_size': agent.state_size,
            'action_size': agent.action_size
        }
        
        metadata_path = model_path.replace('.h5', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        if is_best:
            print(f"   🏆 Best portfolio model saved!")
    
    def _plot_portfolio_progress(self, rewards, returns, val_scores):
        """Plot training progress"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Rewards
        ax1.plot(rewards, alpha=0.7, label='Episode Rewards')
        if len(rewards) > 50:
            smoothed = pd.Series(rewards).rolling(50).mean()
            ax1.plot(smoothed, color='red', linewidth=2, label='50-ep Average')
        ax1.set_title('Portfolio Training Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Returns
        ax2.plot(returns, alpha=0.7, label='Portfolio Returns')
        if len(returns) > 50:
            smoothed = pd.Series(returns).rolling(50).mean()
            ax2.plot(smoothed, color='green', linewidth=2, label='50-ep Average')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(y=self.profit_target_pct*100, color='green', linestyle='--', alpha=0.7, label='Target')
        ax2.axhline(y=-self.max_loss_pct*100, color='red', linestyle='--', alpha=0.7, label='Max Loss')
        ax2.set_title('Portfolio Returns (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Validation
        if val_scores:
            episodes = list(range(200, len(val_scores) * 200 + 1, 200))
            ax3.plot(episodes, val_scores, 'o-', color='orange', label='Validation Score')
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.axhline(y=self.profit_target_pct*100, color='green', linestyle='--', alpha=0.7)
            ax3.axhline(y=-self.max_loss_pct*100, color='red', linestyle='--', alpha=0.7)
            ax3.set_title('Validation Performance')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Validation Return %')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Portfolio composition over time (show diversification)
        ax4.text(0.5, 0.5, f'Portfolio Stocks:\n{", ".join(self.symbols)}\n\nDiversification Strategy:\n• Max {25}% per stock\n• Multi-stock decisions\n• Risk-adjusted rewards', 
                transform=ax4.transAxes, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax4.set_title('Portfolio Configuration')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig('portfolio_models/portfolio_training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()

# Training script for portfolio agent
if __name__ == "__main__":
    print("🎯 Multi-Stock Portfolio Trading RL Training")
    print("=" * 60)
    
    # Create portfolio trainer with multiple stocks
    trainer = PortfolioTradingTrainer(
        symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA'],  # 6-stock portfolio
        start_date='2019-01-01',
        end_date='2024-01-01',
        initial_balance=10000,        # $10k starting capital (more for diversification)
        max_loss_pct=0.30,           # 30% max loss
        profit_target_pct=0.50       # 50% profit target
    )
    
    # Train the portfolio agent
    agent, history = trainer.train_portfolio_agent(
        episodes=2000,               # More episodes for complex portfolio decisions
        save_frequency=200           # Save every 200 episodes
    )
    
    print("\n🎉 Portfolio model training completed!")
    print("📁 Check 'portfolio_models' folder for saved models")
    print("🔄 The agent can now make diversified trading decisions across multiple stocks!")