import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras import models
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedPortfolioTester:
    """Enhanced tester with debugging and forced trading modes"""
    
    def __init__(self, model_path, metadata_path):
        """Initialize with trained model"""
        print("🔄 Loading trained portfolio model...")
        
        # Load model
        self.model = models.load_model(model_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.symbols = self.metadata['symbols']
        self.initial_balance = self.metadata['initial_balance']
        self.max_loss_pct = self.metadata['max_loss_pct']
        self.profit_target_pct = self.metadata['profit_target_pct']
        
        print(f"✅ Model loaded successfully!")
        print(f"   • Stocks: {self.symbols}")
        print(f"   • Initial balance: ${self.initial_balance}")
        print(f"   • Training episode: {self.metadata.get('episode', 'Unknown')}")
        print(f"   • Validation score: {self.metadata.get('validation_score', 'Unknown'):.2f}%")
    
    def analyze_model_behavior(self, days_back=30):
        """Analyze what the model wants to do"""
        print(f"\n🔍 ANALYZING MODEL BEHAVIOR...")
        
        # Get fresh data
        test_data = self.fetch_test_data(days_back=days_back)
        env = DebugPortfolioEnvironment(test_data, self.initial_balance)
        
        state = env.reset()
        action_counts = {i: 0 for i in range(30)}  # 6 stocks × 5 actions
        action_preferences = []
        
        for day in range(min(20, len(list(test_data.values())[0]))):  # First 20 days
            # Get model predictions
            q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
            
            # Analyze top 5 preferred actions
            top_actions = np.argsort(q_values)[-5:][::-1]  # Top 5 actions
            
            action_info = []
            for action_idx in top_actions:
                stock_idx = action_idx // 5
                action_type = action_idx % 5
                symbol = self.symbols[stock_idx] if stock_idx < len(self.symbols) else "UNKNOWN"
                action_names = ['Hold', 'Buy', 'Sell', 'Buy More', 'Sell Half']
                action_name = action_names[action_type]
                q_value = q_values[action_idx]
                
                action_info.append({
                    'symbol': symbol,
                    'action': action_name,
                    'q_value': q_value,
                    'action_idx': action_idx
                })
                action_counts[action_idx] += 1
            
            action_preferences.append(action_info)
            
            # Step environment
            chosen_action = np.argmax(q_values)
            state, _, done, _ = env.step(chosen_action)
            
            if done:
                break
        
        # Print analysis
        print(f"\n📊 MODEL DECISION ANALYSIS:")
        print(f"=" * 50)
        
        # Most preferred actions
        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"🎯 Most Preferred Actions (Top 10):")
        for i, (action_idx, count) in enumerate(sorted_actions[:10]):
            stock_idx = action_idx // 5
            action_type = action_idx % 5
            symbol = self.symbols[stock_idx] if stock_idx < len(self.symbols) else "UNKNOWN"
            action_names = ['Hold', 'Buy', 'Sell', 'Buy More', 'Sell Half']
            action_name = action_names[action_type]
            print(f"   {i+1:2d}. {symbol:4s} {action_name:8s} - chosen {count:2d} times")
        
        # Daily preferences
        print(f"\n📅 DAILY TOP PREFERENCES (First 10 days):")
        for day, day_actions in enumerate(action_preferences[:10]):
            print(f"Day {day:2d}: {day_actions[0]['symbol']:4s} {day_actions[0]['action']:8s} "
                  f"(Q={day_actions[0]['q_value']:.3f})")
        
        return action_preferences, action_counts
    
    def run_aggressive_backtest(self, days_back=60, force_buy_percentage=0.1):
        """Run backtest with forced buying to overcome conservative behavior"""
        print(f"\n🚀 RUNNING AGGRESSIVE BACKTEST...")
        print(f"   • Force buy threshold: {force_buy_percentage*100}% chance per day")
        
        test_data = self.fetch_test_data(days_back=days_back)
        env = AggressivePortfolioEnvironment(test_data, self.initial_balance, force_buy_percentage)
        
        state = env.reset()
        
        portfolio_values = [self.initial_balance]
        trade_log = []
        
        day = 0
        while True:
            # Get model prediction
            q_values = self.model.predict(state.reshape(1, -1), verbose=0)
            model_action = np.argmax(q_values)
            
            # Environment decides whether to force action or use model
            actual_action, was_forced = env.decide_action(model_action)
            
            # Decode action for logging
            stock_idx = actual_action // 5
            action_type = actual_action % 5
            symbol = self.symbols[stock_idx] if stock_idx < len(self.symbols) else self.symbols[0]
            action_names = ['Hold', 'Buy', 'Sell', 'Buy More', 'Sell Half']
            action_name = action_names[action_type]
            
            # Execute action
            next_state, reward, done, info = env.step(actual_action)
            
            portfolio_values.append(info['portfolio_value'])
            
            # Log significant actions
            if action_type != 0 or was_forced:  # Not hold or was forced
                trade_log.append({
                    'day': day,
                    'stock': symbol,
                    'action': action_name,
                    'was_forced': was_forced,
                    'portfolio_value': info['portfolio_value'],
                    'balance': info['balance'],
                    'total_trades': info['total_trades']
                })
            
            if day % 10 == 0:
                forced_str = " (FORCED)" if was_forced else ""
                print(f"   Day {day:3d}: {symbol:4s} {action_name:8s}{forced_str:9s} | "
                      f"Portfolio: ${info['portfolio_value']:8,.0f} | "
                      f"Return: {info['return_pct']:6.2f}% | "
                      f"Trades: {info['total_trades']:3d}")
            
            state = next_state
            day += 1
            
            if done:
                break
        
        # Results
        final_value = info['portfolio_value']
        total_return = (final_value - self.initial_balance) / self.initial_balance * 100
        
        results = {
            'initial_balance': self.initial_balance,
            'final_value': final_value,
            'total_return_pct': total_return,
            'total_trades': info['total_trades'],
            'days_traded': day,
            'portfolio_values': portfolio_values,
            'trade_log': trade_log,
            'final_positions': info['positions'],
            'symbols': self.symbols,
            'forced_trades': sum(1 for t in trade_log if t['was_forced'])
        }
        
        self._print_aggressive_results(results)
        return results
    
    def _print_aggressive_results(self, results):
        """Print aggressive backtest results"""
        print(f"\n📊 AGGRESSIVE BACKTEST RESULTS")
        print("=" * 50)
        print(f"💰 Financial Performance:")
        print(f"   • Initial Balance:    ${results['initial_balance']:,.0f}")
        print(f"   • Final Value:        ${results['final_value']:,.0f}")
        print(f"   • Total Return:       {results['total_return_pct']:+.2f}%")
        print(f"   • Days Traded:        {results['days_traded']}")
        print(f"   • Total Trades:       {results['total_trades']}")
        print(f"   • Forced Trades:      {results['forced_trades']}")
        print(f"   • Model Trades:       {results['total_trades'] - results['forced_trades']}")
        
        # Position breakdown
        print(f"\n🏦 Final Portfolio Breakdown:")
        balance = results['final_positions']['balance']
        print(f"   • Cash Balance:       ${balance:,.0f}")
        
        total_stock_value = 0
        for symbol, pos_info in results['final_positions']['positions'].items():
            if pos_info['value'] > 0:
                print(f"   • {symbol:4s} Position:     ${pos_info['value']:8,.0f} "
                      f"({pos_info['percentage']:5.1f}%) - {pos_info['shares']} shares")
                total_stock_value += pos_info['value']
        
        cash_pct = balance / results['final_value'] * 100 if results['final_value'] > 0 else 100
        stocks_pct = total_stock_value / results['final_value'] * 100 if results['final_value'] > 0 else 0
        
        print(f"\n💼 Asset Allocation:")
        print(f"   • Cash:               {cash_pct:.1f}%")
        print(f"   • Stocks:             {stocks_pct:.1f}%")
        
        if results['forced_trades'] > 0:
            print(f"\n⚡ Forced Trading Analysis:")
            print(f"   • Without forced trades, model would have made {results['total_trades'] - results['forced_trades']} trades")
            print(f"   • Forced trades helped overcome conservative behavior")
    
    def fetch_test_data(self, start_date=None, end_date=None, days_back=90):
        """Fetch fresh data for testing"""
        if start_date is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
        
        stock_data_dict = {}
        for symbol in self.symbols:
            try:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if len(data) > 10:
                    stock_data_dict[symbol] = data.dropna()
                else:
                    print(f"   ⚠️ {symbol}: Insufficient data ({len(data)} days)")
            except Exception as e:
                print(f"   ❌ {symbol}: {e}")
        
        # Align data
        min_length = min(len(data) for data in stock_data_dict.values())
        for symbol in stock_data_dict.keys():
            stock_data_dict[symbol] = stock_data_dict[symbol].iloc[:min_length].reset_index(drop=True)
        
        return stock_data_dict
    
    def create_training_comparison(self):
        """Compare current market vs training period performance"""
        print(f"\n📈 MARKET COMPARISON ANALYSIS...")
        
        # Get current market data (last 90 days)
        current_data = self.fetch_test_data(days_back=90)
        
        # Get training period data (approximate)
        training_start = datetime(2019, 1, 1)
        training_end = datetime(2024, 1, 1)
        
        print(f"📊 Current Market (last 90 days):")
        current_returns = {}
        for symbol in self.symbols:
            if symbol in current_data and len(current_data[symbol]) > 0:
                try:
                    data = current_data[symbol]
                    # Extract scalar values from Series
                    start_price = float(data['Close'].iloc[0])
                    end_price = float(data['Close'].iloc[-1])
                    return_pct = (end_price - start_price) / start_price * 100
                    current_returns[symbol] = return_pct
                    print(f"   • {symbol}: {return_pct:+6.2f}%")
                except Exception as e:
                    print(f"   ⚠️ {symbol}: Error calculating return - {e}")
        
        if current_returns:
            avg_current = sum(current_returns.values()) / len(current_returns)
            print(f"   📊 Average current return: {avg_current:+6.2f}%")
            
            print(f"\n🤔 ANALYSIS:")
            if avg_current > 5:
                print(f"   • Current market is performing well (+{avg_current:.1f}%)")
                print(f"   • Model's conservative behavior is missing opportunities")
            elif avg_current < -5:
                print(f"   • Current market is declining ({avg_current:.1f}%)")
                print(f"   • Model's cash position might be protecting capital")
            else:
                print(f"   • Current market is relatively flat ({avg_current:.1f}%)")
                print(f"   • Model behavior is reasonable for sideways market")
        else:
            print(f"   ⚠️ No valid return data available for analysis")


class DebugPortfolioEnvironment:
    """Debug environment that doesn't actually trade, just tracks decisions"""
    
    def __init__(self, stock_data_dict, initial_balance):
        self.stock_data_dict = stock_data_dict
        self.symbols = list(stock_data_dict.keys())
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = {symbol: 0 for symbol in self.symbols}
        return self._get_observation()
    
    def _get_observation(self):
        """Simplified observation"""
        if self.current_step >= len(list(self.stock_data_dict.values())[0]):
            return np.zeros(4 + len(self.symbols) * 8)
        
        # Simple portfolio features
        portfolio_features = [1.0, 0.0, 0.0, 0.0]  # All cash, no positions
        
        # Simple stock features
        stock_features = []
        for symbol in self.symbols:
            if self.current_step < len(self.stock_data_dict[symbol]):
                price = float(self.stock_data_dict[symbol].iloc[self.current_step]['Close'])
                features = [price/100, 0.5, price/100, price/100, 0.5, 0, 0, 0]  # Normalized
                stock_features.extend(features)
            else:
                stock_features.extend([0] * 8)
        
        return np.array(portfolio_features + stock_features, dtype=np.float32)
    
    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(list(self.stock_data_dict.values())[0]) - 1
        
        info = {
            'portfolio_value': self.initial_balance,
            'balance': self.initial_balance,
            'total_trades': 0,
            'return_pct': 0.0,
            'positions': {'balance': self.initial_balance, 'positions': {s: {'shares': 0, 'value': 0, 'percentage': 0} for s in self.symbols}}
        }
        
        return self._get_observation(), 0, done, info


class AggressivePortfolioEnvironment:
    """Environment that forces some trading to overcome conservative model"""
    
    def __init__(self, stock_data_dict, initial_balance, force_buy_probability=0.1):
        self.stock_data_dict = stock_data_dict
        self.symbols = list(stock_data_dict.keys())
        self.initial_balance = initial_balance
        self.force_buy_probability = force_buy_probability
        self.max_position_pct = 0.3  # Allow 30% per stock
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = {symbol: 0 for symbol in self.symbols}
        self.total_trades = 0
        self.forced_actions = 0
        return self._get_observation()
    
    def decide_action(self, model_action):
        """Decide whether to use model action or force a buy"""
        # If model wants to hold and we have lots of cash, sometimes force a buy
        stock_idx = model_action // 5
        action_type = model_action % 5
        
        cash_ratio = self.balance / self._calculate_portfolio_value()
        
        # Force buy if:
        # 1. Model wants to hold (action_type == 0)
        # 2. We have >80% cash
        # 3. Random chance
        if (action_type == 0 and 
            cash_ratio > 0.8 and 
            np.random.random() < self.force_buy_probability):
            
            # Force buy action for a random stock
            forced_stock_idx = np.random.randint(0, len(self.symbols))
            forced_action = forced_stock_idx * 5 + 1  # Buy action
            return forced_action, True
        
        return model_action, False
    
    def _get_observation(self):
        """Simplified observation for testing"""
        if self.current_step >= len(list(self.stock_data_dict.values())[0]):
            return np.zeros(4 + len(self.symbols) * 8)
        
        # Portfolio features
        portfolio_value = self._calculate_portfolio_value()
        balance_ratio = self.balance / self.initial_balance
        portfolio_return = (portfolio_value - self.initial_balance) / self.initial_balance
        
        current_prices = self._get_current_prices()
        position_values = [self.shares_held[symbol] * current_prices[symbol] for symbol in self.symbols]
        total_invested = sum(position_values)
        
        if total_invested > 0:
            weights = [val / total_invested for val in position_values]
            diversification_ratio = 1 - sum(w**2 for w in weights)
        else:
            diversification_ratio = 0
        
        risk_exposure = total_invested / portfolio_value if portfolio_value > 0 else 0
        
        portfolio_features = [balance_ratio, portfolio_return, diversification_ratio, risk_exposure]
        
        # Stock features
        stock_features = []
        for symbol in self.symbols:
            if self.current_step < len(self.stock_data_dict[symbol]):
                price = float(self.stock_data_dict[symbol].iloc[self.current_step]['Close'])
                volume = float(self.stock_data_dict[symbol].iloc[self.current_step]['Volume']) / 1000000
                holdings = self.shares_held[symbol]
                position_value = holdings * price
                
                features = [
                    price / 100, volume, price / 100, price / 100, 0.5,
                    holdings / 100, position_value / self.initial_balance, 0
                ]
                stock_features.extend(features)
            else:
                stock_features.extend([0] * 8)
        
        return np.array(portfolio_features + stock_features, dtype=np.float32)
    
    def _get_current_prices(self):
        if self.current_step >= len(list(self.stock_data_dict.values())[0]):
            last_step = len(list(self.stock_data_dict.values())[0]) - 1
            return {symbol: float(data.iloc[last_step]['Close']) 
                   for symbol, data in self.stock_data_dict.items()}
        return {symbol: float(data.iloc[self.current_step]['Close']) 
                for symbol, data in self.stock_data_dict.items()}
    
    def _calculate_portfolio_value(self):
        current_prices = self._get_current_prices()
        stock_values = sum(self.shares_held[symbol] * current_prices[symbol] 
                          for symbol in self.symbols)
        return self.balance + stock_values
    
    def step(self, action):
        if self.current_step >= len(list(self.stock_data_dict.values())[0]) - 1:
            return self._get_observation(), 0, True, self._get_info()
        
        # Decode action
        stock_idx = action // 5
        action_type = action % 5
        if stock_idx >= len(self.symbols):
            stock_idx = 0
        symbol = self.symbols[stock_idx]
        
        current_prices = self._get_current_prices()
        current_price = current_prices[symbol]
        portfolio_value = self._calculate_portfolio_value()
        
        # Execute actions more aggressively
        if action_type == 1 or action_type == 3:  # Buy or Buy More
            max_investment = min(
                self.balance * 0.9,  # Use 90% of cash
                portfolio_value * self.max_position_pct  # Position limit
            )
            
            if max_investment > current_price * 1.001:
                shares_to_buy = int(max_investment / (current_price * 1.001))
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * 1.001
                    self.balance -= cost
                    self.shares_held[symbol] += shares_to_buy
                    self.total_trades += 1
        
        elif action_type == 2 and self.shares_held[symbol] > 0:  # Sell
            sell_value = self.shares_held[symbol] * current_price * 0.999
            self.balance += sell_value
            self.shares_held[symbol] = 0
            self.total_trades += 1
        
        elif action_type == 4 and self.shares_held[symbol] > 1:  # Sell Half
            shares_to_sell = self.shares_held[symbol] // 2
            sell_value = shares_to_sell * current_price * 0.999
            self.balance += sell_value
            self.shares_held[symbol] -= shares_to_sell
            self.total_trades += 1
        
        self.current_step += 1
        done = (self.current_step >= len(list(self.stock_data_dict.values())[0]) - 1)
        
        return self._get_observation(), 0, done, self._get_info()
    
    def _get_info(self):
        portfolio_value = self._calculate_portfolio_value()
        current_prices = self._get_current_prices()
        
        positions = {}
        for symbol in self.symbols:
            position_value = self.shares_held[symbol] * current_prices[symbol]
            positions[symbol] = {
                'shares': self.shares_held[symbol],
                'value': position_value,
                'percentage': position_value / portfolio_value * 100 if portfolio_value > 0 else 0
            }
        
        return {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'total_trades': self.total_trades,
            'return_pct': (portfolio_value - self.initial_balance) / self.initial_balance * 100,
            'positions': {'balance': self.balance, 'positions': positions}
        }


def find_best_model(models_dir="portfolio_models"):
    """Find the best saved model"""
    if not os.path.exists(models_dir):
        print(f"❌ Models directory '{models_dir}' not found!")
        return None, None
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
    if not model_files:
        print("❌ No model files found!")
        return None, None
    
    # Prefer 'best' models, otherwise use most recent
    best_files = [f for f in model_files if 'best' in f]
    if best_files:
        best_files.sort(reverse=True)
        model_file = best_files[0]
    else:
        model_files.sort(reverse=True)
        model_file = model_files[0]
    
    model_path = os.path.join(models_dir, model_file)
    metadata_path = model_path.replace('.h5', '_metadata.pkl')
    
    if not os.path.exists(metadata_path):
        metadata_files = [f for f in os.listdir(models_dir) if f.endswith('_metadata.pkl')]
        if metadata_files:
            metadata_files.sort(reverse=True)
            metadata_path = os.path.join(models_dir, metadata_files[0])
        else:
            print("❌ No metadata files found!")
            return None, None
    
    return model_path, metadata_path


# Enhanced testing script
if __name__ == "__main__":
    print("🔧 Enhanced Portfolio Model Debugging")
    print("=" * 60)
    
    # Find model
    model_path, metadata_path = find_best_model()
    
    if model_path and metadata_path:
        try:
            # Create enhanced tester
            tester = EnhancedPortfolioTester(model_path, metadata_path)
            
            # Step 1: Analyze model behavior
            print(f"\n🔍 STEP 1: Analyze Model Decision Patterns")
            tester.analyze_model_behavior(days_back=30)
            
            # Step 2: Market comparison
            print(f"\n📈 STEP 2: Current Market Analysis")
            tester.create_training_comparison()
            
            # Step 3: Aggressive backtest
            print(f"\n⚡ STEP 3: Force Trading Mode")
            results = tester.run_aggressive_backtest(days_back=60, force_buy_percentage=0.15)
            
            print(f"\n✅ Enhanced testing completed!")
            print(f"\n💡 RECOMMENDATIONS:")
            if results['total_return_pct'] > 5:
                print(f"   ✅ Model shows potential when forced to trade")
                print(f"   🎯 Consider retraining with less conservative parameters")
            else:
                print(f"   ⚠️ Model may need retraining with different reward structure")
                print(f"   🔄 Try longer training or adjust risk parameters")
            
        except Exception as e:
            print(f"❌ Enhanced testing failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Could not find trained model files.")