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

# Fix pandas issue
pd.options.mode.chained_assignment = None

class StockTradingEnvironment(gym.Env):
    """Custom Stock Trading Environment for RL Training"""
    
    def __init__(self, stock_data, initial_balance=1000, transaction_cost=0.001, 
                 max_loss_pct=0.40, profit_target_pct=0.30):
        super(StockTradingEnvironment, self).__init__()
        
        self.stock_data = stock_data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_loss_pct = max_loss_pct
        self.profit_target_pct = profit_target_pct
        
        # Risk thresholds
        self.stop_loss_threshold = initial_balance * (1 - max_loss_pct)
        self.profit_target = initial_balance * (1 + profit_target_pct)
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space with risk features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(12,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_portfolio_value = self.initial_balance
        self.trades_made = 0
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current state observation"""
        if self.current_step >= len(self.stock_data):
            return np.zeros(12)
        
        row = self.stock_data.iloc[self.current_step]
        current_price = float(row['Close'])
        
        # Technical indicators
        sma_5 = float(self._get_sma(5))
        sma_20 = float(self._get_sma(20))
        rsi = float(self._get_rsi())
        
        # Risk metrics
        current_loss_pct = max(0, (self.initial_balance - self.net_worth) / self.initial_balance)
        current_profit_pct = max(0, (self.net_worth - self.initial_balance) / self.initial_balance)
        
        # Safe division for price ratios
        price_sma5_ratio = (current_price - sma_5) / current_price if sma_5 > 0 and current_price > 0 else 0
        price_sma20_ratio = (current_price - sma_20) / current_price if sma_20 > 0 and current_price > 0 else 0
        
        obs = np.array([
            self.balance / self.initial_balance,
            self.shares_held,
            current_price,
            float(row['Volume']) / 1000000,
            sma_5,
            sma_20,
            rsi,
            price_sma5_ratio,
            price_sma20_ratio,
            self.net_worth / self.initial_balance,
            current_loss_pct,
            current_profit_pct
        ])
        
        return obs.astype(np.float32)
    
    def _get_sma(self, window):
        """Simple Moving Average - returns float"""
        if self.current_step < window:
            return float(self.stock_data.iloc[0]['Close'])
        
        start_idx = max(0, self.current_step - window + 1)
        sma_value = self.stock_data.iloc[start_idx:self.current_step + 1]['Close'].mean()
        return float(sma_value)
    
    def _get_rsi(self, window=14):
        """Relative Strength Index - returns float"""
        if self.current_step < window:
            return 50.0
        
        start_idx = max(0, self.current_step - window)
        prices = self.stock_data.iloc[start_idx:self.current_step + 1]['Close']
        
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
        
        # Ensure we return a valid float
        if pd.isna(rsi) or np.isnan(rsi):
            return 50.0
        
        return float(rsi)
    
    def step(self, action):
        if self.current_step >= len(self.stock_data) - 1:
            return self._get_observation(), 0, True, {}
        
        current_price = float(self.stock_data.iloc[self.current_step]['Close'])
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Update max portfolio value
        if self.net_worth > self.max_portfolio_value:
            self.max_portfolio_value = self.net_worth
        
        # Risk management: Force actions if needed
        if self.net_worth <= self.stop_loss_threshold and self.shares_held > 0:
            action = 2  # Force sell
        elif self.net_worth >= self.profit_target and self.shares_held > 0:
            action = 2  # Force sell
        
        # Execute action
        reward = 0
        
        if action == 1:  # Buy
            if self.net_worth > self.stop_loss_threshold * 1.1:
                available_balance = self.balance * 0.8
                shares_to_buy = int(available_balance // (current_price * (1 + self.transaction_cost)))
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                    self.balance -= cost
                    self.shares_held += shares_to_buy
                    self.trades_made += 1
        
        elif action == 2 and self.shares_held > 0:  # Sell
            sell_value = self.shares_held * current_price * (1 - self.transaction_cost)
            self.balance += sell_value
            self.shares_held = 0
            self.trades_made += 1
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        if self.current_step < len(self.stock_data):
            new_price = float(self.stock_data.iloc[self.current_step]['Close'])
            self.net_worth = self.balance + self.shares_held * new_price
            
            # Portfolio return based reward
            portfolio_return = (self.net_worth - self.initial_balance) / self.initial_balance
            reward = portfolio_return * 100
            
            # Risk adjustments
            if self.net_worth <= self.stop_loss_threshold:
                reward -= 50
            elif self.net_worth >= self.profit_target:
                reward += 20
            
            # Anti-overtrading
            if self.trades_made > 50:
                reward -= 0.5 * (self.trades_made - 50)
        
        done = (self.current_step >= len(self.stock_data) - 1 or 
                self.net_worth <= self.stop_loss_threshold or 
                self.net_worth >= self.profit_target)
        
        info = {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'trades_made': self.trades_made,
            'return_pct': (self.net_worth - self.initial_balance) / self.initial_balance * 100
        }
        
        return self._get_observation(), reward, done, info

class DQNAgent:
    """Deep Q-Network Agent for Stock Trading"""
    
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # Increased memory
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.gamma = 0.95  # Discount factor
        
        # Build networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
        
        # Training stats
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'epsilon': [],
            'losses': []
        }
    
    def _build_model(self):
        """Build the neural network"""
        model = models.Sequential([
            layers.Dense(128, input_dim=self.state_size, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return model
    
    def update_target_network(self):
        """Copy weights to target network"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Epsilon-greedy action selection"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size=64):
        """Train the model"""
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
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss

class StockTradingTrainer:
    """Stock Trading Model Trainer"""
    
    def __init__(self, symbols=['AAPL'], start_date='2019-01-01', end_date='2024-01-01',
                 initial_balance=1000, max_loss_pct=0.40, profit_target_pct=0.30):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.max_loss_pct = max_loss_pct
        self.profit_target_pct = profit_target_pct
        
        # Create models directory
        self.models_dir = "trained_models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        print(f"🎯 Training Configuration:")
        print(f"   • Initial Balance: ${initial_balance}")
        print(f"   • Max Loss: {max_loss_pct*100}% (${initial_balance * max_loss_pct})")
        print(f"   • Profit Target: {profit_target_pct*100}% (${initial_balance * profit_target_pct})")
    
    def fetch_training_data(self):
        """Fetch and prepare training data"""
        print(f"\n📊 Fetching training data for {self.symbols}...")
        
        all_data = []
        for symbol in self.symbols:
            try:
                data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
                if len(data) > 0:
                    data['Symbol'] = symbol
                    all_data.append(data)
                    print(f"   ✅ {symbol}: {len(data)} days")
            except Exception as e:
                print(f"   ❌ {symbol}: {e}")
        
        if not all_data:
            raise Exception("No data could be fetched")
        
        # Use first symbol's data for training
        self.stock_data = all_data[0].dropna()
        
        # Split into training and validation
        split_point = int(len(self.stock_data) * 0.85)
        self.train_data = self.stock_data.iloc[:split_point]
        self.validation_data = self.stock_data.iloc[split_point:]
        
        print(f"   📈 Total data: {len(self.stock_data)} days")
        print(f"   🏋️ Training: {len(self.train_data)} days")
        print(f"   ✅ Validation: {len(self.validation_data)} days")
        
        return self.train_data, self.validation_data
    
    def create_agent(self):
        """Create and return DQN agent"""
        # Create environment to get state/action dimensions
        temp_env = StockTradingEnvironment(
            self.train_data, 
            self.initial_balance,
            max_loss_pct=self.max_loss_pct,
            profit_target_pct=self.profit_target_pct
        )
        
        state_size = temp_env.observation_space.shape[0]
        action_size = temp_env.action_space.n
        
        agent = DQNAgent(state_size, action_size)
        print(f"\n🤖 Agent created:")
        print(f"   • State size: {state_size}")
        print(f"   • Action size: {action_size}")
        print(f"   • Memory capacity: {agent.memory.maxlen}")
        
        return agent
    
    def train_agent(self, episodes=1000, save_frequency=100):
        """Train the RL agent"""
        print(f"\n🚀 Starting training for {episodes} episodes...")
        
        # Fetch data and create agent
        train_data, validation_data = self.fetch_training_data()
        agent = self.create_agent()
        
        # Create training environment
        env = StockTradingEnvironment(
            train_data,
            self.initial_balance,
            max_loss_pct=self.max_loss_pct,
            profit_target_pct=self.profit_target_pct
        )
        
        # Training metrics
        episode_rewards = []
        episode_returns = []
        validation_scores = []
        losses = []
        
        best_validation_score = float('-inf')
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            episode_loss = 0
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
                    episode_returns.append(info['return_pct'])
                    break
            
            # Train the agent
            if len(agent.memory) > 128:
                loss = agent.replay(batch_size=64)
                episode_loss = loss
                losses.append(loss)
            
            # Update target network
            if episode % 50 == 0:
                agent.update_target_network()
            
            # Validation and saving
            if episode % save_frequency == 0 and episode > 0:
                # Run validation
                val_score = self._validate_agent(agent, validation_data)
                validation_scores.append(val_score)
                
                # Save if best model
                if val_score > best_validation_score:
                    best_validation_score = val_score
                    self._save_model(agent, episode, val_score, is_best=True)
                
                # Regular save
                self._save_model(agent, episode, val_score, is_best=False)
                
                # Progress report
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                avg_return = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else np.mean(episode_returns)
                avg_loss = np.mean(losses[-10:]) if losses else 0
                
                print(f"Episode {episode:4d} | "
                      f"Reward: {avg_reward:8.2f} | "
                      f"Return: {avg_return:6.2f}% | "
                      f"Val Score: {val_score:6.2f}% | "
                      f"Loss: {avg_loss:.4f} | "
                      f"ε: {agent.epsilon:.3f}")
        
        # Save final model
        final_val_score = self._validate_agent(agent, validation_data)
        self._save_model(agent, episodes, final_val_score, is_best=False, is_final=True)
        
        # Save training history
        training_history = {
            'episode_rewards': episode_rewards,
            'episode_returns': episode_returns,
            'validation_scores': validation_scores,
            'losses': losses,
            'config': {
                'symbols': self.symbols,
                'start_date': self.start_date,
                'end_date': self.end_date,
                'initial_balance': self.initial_balance,
                'max_loss_pct': self.max_loss_pct,
                'profit_target_pct': self.profit_target_pct,
                'episodes': episodes
            }
        }
        
        history_file = os.path.join(self.models_dir, 'training_history.pkl')
        with open(history_file, 'wb') as f:
            pickle.dump(training_history, f)
        
        print(f"\n✅ Training completed!")
        print(f"   📁 Models saved in: {self.models_dir}")
        print(f"   🏆 Best validation score: {best_validation_score:.2f}%")
        print(f"   📊 Final validation score: {final_val_score:.2f}%")
        
        # Plot training progress
        self._plot_training_progress(episode_rewards, episode_returns, validation_scores, losses)
        
        return agent, training_history
    
    def _validate_agent(self, agent, validation_data):
        """Validate agent on unseen data"""
        val_env = StockTradingEnvironment(
            validation_data,
            self.initial_balance,
            max_loss_pct=self.max_loss_pct,
            profit_target_pct=self.profit_target_pct
        )
        
        state = val_env.reset()
        
        # Disable exploration for validation
        original_epsilon = agent.epsilon
        agent.epsilon = 0
        
        while True:
            action = agent.act(state)
            state, reward, done, info = val_env.step(action)
            
            if done:
                validation_return = info['return_pct']
                break
        
        # Restore exploration
        agent.epsilon = original_epsilon
        
        return validation_return
    
    def _save_model(self, agent, episode, val_score, is_best=False, is_final=False):
        """Save the trained model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_best:
            filename = f"best_model_{timestamp}.h5"
        elif is_final:
            filename = f"final_model_{timestamp}.h5"
        else:
            filename = f"model_ep{episode}_{timestamp}.h5"
        
        model_path = os.path.join(self.models_dir, filename)
        agent.q_network.save(model_path)
        
        # Save metadata
        metadata = {
            'episode': episode,
            'validation_score': val_score,
            'epsilon': agent.epsilon,
            'symbols': self.symbols,
            'initial_balance': self.initial_balance,
            'max_loss_pct': self.max_loss_pct,
            'profit_target_pct': self.profit_target_pct,
            'timestamp': timestamp
        }
        
        metadata_path = os.path.join(self.models_dir, filename.replace('.h5', '_metadata.pkl'))
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        if is_best:
            print(f"   🏆 Best model saved: {filename}")
        elif is_final:
            print(f"   💾 Final model saved: {filename}")
    
    def _plot_training_progress(self, rewards, returns, val_scores, losses):
        """Plot training progress"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        ax1.plot(rewards, alpha=0.7)
        if len(rewards) > 50:
            smoothed = pd.Series(rewards).rolling(50).mean()
            ax1.plot(smoothed, color='red', linewidth=2)
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        
        # Portfolio returns
        ax2.plot(returns, alpha=0.7)
        if len(returns) > 50:
            smoothed = pd.Series(returns).rolling(50).mean()
            ax2.plot(smoothed, color='green', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(y=self.profit_target_pct*100, color='green', linestyle='--', alpha=0.7, label='Profit Target')
        ax2.axhline(y=-self.max_loss_pct*100, color='red', linestyle='--', alpha=0.7, label='Max Loss')
        ax2.set_title('Portfolio Returns (%)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Return %')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Validation scores
        if val_scores:
            episodes = list(range(100, len(val_scores) * 100 + 1, 100))
            ax3.plot(episodes, val_scores, 'o-', color='orange')
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.axhline(y=self.profit_target_pct*100, color='green', linestyle='--', alpha=0.7)
            ax3.axhline(y=-self.max_loss_pct*100, color='red', linestyle='--', alpha=0.7)
            ax3.set_title('Validation Performance')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Validation Return %')
            ax3.grid(True, alpha=0.3)
        
        # Training losses
        if losses:
            ax4.plot(losses, alpha=0.7)
            if len(losses) > 20:
                smoothed = pd.Series(losses).rolling(20).mean()
                ax4.plot(smoothed, color='purple', linewidth=2)
            ax4.set_title('Training Loss')
            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('Loss')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
        plt.show()

# Training script
if __name__ == "__main__":
    print("🎯 Stock Trading RL Model Training")
    print("=" * 50)
    
    # Create trainer
    trainer = StockTradingTrainer(
        symbols=['AAPL'],           # Stock to trade
        start_date='2019-01-01',    # Training data start
        end_date='2024-01-01',      # Training data end
        initial_balance=1000,       # $1000 starting capital
        max_loss_pct=0.40,          # 40% max loss
        profit_target_pct=0.30      # 30% profit target
    )
    
    # Train the model
    agent, history = trainer.train_agent(
        episodes=1500,              # Number of training episodes
        save_frequency=100          # Save model every 100 episodes
    )
    
    print("\n🎉 Model training completed!")
    print(f"📁 Check the 'trained_models' folder for saved models")
    print(f"🔄 Next step: Use the trained model for predictions and live trading")