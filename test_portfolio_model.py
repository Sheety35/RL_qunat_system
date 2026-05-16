import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras import models
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent hanging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PortfolioModelTester:
    """Test trained portfolio models on new data"""
    
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
    
    def fetch_test_data(self, start_date=None, end_date=None, days_back=90):
        """Fetch fresh data for testing"""
        if start_date is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
        
        print(f"\n📊 Fetching test data ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})...")
        
        stock_data_dict = {}
        for symbol in self.symbols:
            try:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if len(data) > 10:  # Need sufficient data
                    stock_data_dict[symbol] = data.dropna()
                    print(f"   ✅ {symbol}: {len(data)} days")
                else:
                    print(f"   ⚠️ {symbol}: Insufficient data ({len(data)} days)")
            except Exception as e:
                print(f"   ❌ {symbol}: {e}")
        
        if len(stock_data_dict) < 2:
            raise Exception("Need at least 2 stocks with valid data")
        
        # Align data to same length
        min_length = min(len(data) for data in stock_data_dict.values())
        for symbol in stock_data_dict.keys():
            stock_data_dict[symbol] = stock_data_dict[symbol].iloc[:min_length].reset_index(drop=True)
        
        print(f"   📈 Test dataset: {min_length} days")
        return stock_data_dict
    
    def create_test_environment(self, stock_data_dict):
        """Create testing environment (simplified version of training env)"""
        return PortfolioTestEnvironment(
            stock_data_dict, 
            self.initial_balance,
            max_loss_pct=self.max_loss_pct,
            profit_target_pct=self.profit_target_pct
        )
    
    def run_backtest(self, test_data=None, days_back=90, verbose=True):
        """Run full backtest on fresh data"""
        print(f"\n🚀 Starting Portfolio Backtest...")
        
        # Fetch data if not provided
        if test_data is None:
            test_data = self.fetch_test_data(days_back=days_back)
        
        # Create test environment
        env = self.create_test_environment(test_data)
        
        # Run the test
        state = env.reset()
        
        # Track performance
        portfolio_values = [self.initial_balance]
        daily_returns = []
        trade_log = []
        positions_over_time = []
        
        day = 0
        while True:
            # Model prediction (no exploration)
            q_values = self.model.predict(state.reshape(1, -1), verbose=0)
            action = np.argmax(q_values[0])
            
            # Decode action
            stock_idx = action // 5
            action_type = action % 5
            symbol = self.symbols[stock_idx] if stock_idx < len(self.symbols) else self.symbols[0]
            action_names = ['Hold', 'Buy', 'Sell', 'Buy More', 'Sell Half']
            action_name = action_names[action_type]
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Log data
            portfolio_values.append(info['portfolio_value'])
            if len(portfolio_values) > 1:
                daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
                daily_returns.append(daily_return)
            
            # Log significant trades
            if action_type != 0:  # Not hold
                trade_log.append({
                    'day': day,
                    'stock': symbol,
                    'action': action_name,
                    'portfolio_value': info['portfolio_value'],
                    'balance': info['balance'],
                    'total_trades': info['total_trades']
                })
            
            # Log positions
            positions_over_time.append({
                'day': day,
                'portfolio_value': info['portfolio_value'],
                'balance': info['balance'],
                'positions': info['positions'].copy()
            })
            
            if verbose and day % 10 == 0:
                print(f"   Day {day:3d}: {symbol:4s} {action_name:8s} | "
                      f"Portfolio: ${info['portfolio_value']:8,.0f} | "
                      f"Return: {info['return_pct']:6.2f}% | "
                      f"Trades: {info['total_trades']:3d}")
            
            state = next_state
            day += 1
            
            if done:
                break
        
        # Calculate performance metrics
        final_value = info['portfolio_value']
        total_return = (final_value - self.initial_balance) / self.initial_balance
        
        results = {
            'initial_balance': self.initial_balance,
            'final_value': final_value,
            'total_return_pct': total_return * 100,
            'total_trades': info['total_trades'],
            'days_traded': day,
            'portfolio_values': portfolio_values,
            'daily_returns': daily_returns,
            'trade_log': trade_log,
            'positions_over_time': positions_over_time,
            'final_positions': info['positions'],
            'symbols': self.symbols
        }
        
        self._print_backtest_summary(results)
        return results
    
    def _print_backtest_summary(self, results):
        """Print detailed backtest results"""
        print(f"\n📊 BACKTEST RESULTS SUMMARY")
        print("=" * 50)
        print(f"💰 Financial Performance:")
        print(f"   • Initial Balance:    ${results['initial_balance']:,.0f}")
        print(f"   • Final Value:        ${results['final_value']:,.0f}")
        print(f"   • Total Return:       {results['total_return_pct']:+.2f}%")
        print(f"   • Days Traded:        {results['days_traded']}")
        print(f"   • Total Trades:       {results['total_trades']}")
        
        # Calculate additional metrics
        if len(results['daily_returns']) > 0:
            daily_returns = np.array(results['daily_returns'])
            volatility = np.std(daily_returns) * np.sqrt(252) * 100  # Annualized
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
            max_dd = self._calculate_max_drawdown(results['portfolio_values'])
            
            print(f"\n📈 Risk Metrics:")
            print(f"   • Daily Volatility:   {np.std(daily_returns)*100:.2f}%")
            print(f"   • Annual Volatility:  {volatility:.2f}%")
            print(f"   • Sharpe Ratio:       {sharpe:.2f}")
            print(f"   • Max Drawdown:       {max_dd:.2f}%")
        
        # Position breakdown
        print(f"\n🏦 Final Portfolio Breakdown:")
        print(f"   • Cash Balance:       ${results['final_positions']['balance']:,.0f}")
        for symbol, pos_info in results['final_positions']['positions'].items():
            if pos_info['value'] > 0:
                print(f"   • {symbol:4s} Position:     ${pos_info['value']:8,.0f} "
                      f"({pos_info['percentage']:5.1f}%) - {pos_info['shares']} shares")
        
        # Trade frequency
        print(f"\n📋 Trading Activity:")
        trades_per_stock = {}
        for trade in results['trade_log']:
            stock = trade['stock']
            trades_per_stock[stock] = trades_per_stock.get(stock, 0) + 1
        
        for symbol in results['symbols']:
            count = trades_per_stock.get(symbol, 0)
            print(f"   • {symbol:4s}: {count:3d} trades")
    
    def _calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown"""
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def plot_backtest_results(self, results, save_path="backtest_results.png"):
        """Create comprehensive visualization of backtest"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        days = range(len(results['portfolio_values']))
        
        # Portfolio value over time
        ax1.plot(days, results['portfolio_values'], linewidth=2, color='green')
        ax1.axhline(y=results['initial_balance'], color='blue', linestyle='--', alpha=0.7, label='Initial Balance')
        ax1.fill_between(days, results['initial_balance'], results['portfolio_values'], 
                        where=np.array(results['portfolio_values']) >= results['initial_balance'], 
                        color='green', alpha=0.3, label='Profit')
        ax1.fill_between(days, results['initial_balance'], results['portfolio_values'], 
                        where=np.array(results['portfolio_values']) < results['initial_balance'], 
                        color='red', alpha=0.3, label='Loss')
        ax1.set_title(f'Portfolio Value Over Time\nTotal Return: {results["total_return_pct"]:+.2f}%')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Daily returns
        if results['daily_returns']:
            ax2.plot(results['daily_returns'], alpha=0.7, color='blue')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_title('Daily Returns')
            ax2.set_xlabel('Days')
            ax2.set_ylabel('Daily Return')
            ax2.grid(True, alpha=0.3)
        
        # Position allocation over time
        if results['positions_over_time']:
            position_data = {}
            for symbol in results['symbols']:
                position_data[symbol] = []
            
            for day_data in results['positions_over_time']:
                for symbol in results['symbols']:
                    pos_pct = day_data['positions'].get(symbol, {}).get('percentage', 0)
                    position_data[symbol].append(pos_pct)
            
            bottom = np.zeros(len(results['positions_over_time']))
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            for i, symbol in enumerate(results['symbols']):
                if any(pos > 0 for pos in position_data[symbol]):
                    ax3.fill_between(range(len(position_data[symbol])), 
                                   bottom, 
                                   bottom + position_data[symbol],
                                   label=symbol, alpha=0.7, color=colors[i % len(colors)])
                    bottom = bottom + np.array(position_data[symbol])
            
            ax3.set_title('Portfolio Allocation Over Time')
            ax3.set_xlabel('Days')
            ax3.set_ylabel('Allocation %')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.grid(True, alpha=0.3)
        
        # Trade timeline
        if results['trade_log']:
            trade_days = [trade['day'] for trade in results['trade_log']]
            trade_values = [trade['portfolio_value'] for trade in results['trade_log']]
            trade_actions = [trade['action'] for trade in results['trade_log']]
            
            # Color code by action
            colors = []
            for action in trade_actions:
                if 'Buy' in action:
                    colors.append('green')
                elif 'Sell' in action:
                    colors.append('red')
                else:
                    colors.append('blue')
            
            ax4.scatter(trade_days, trade_values, c=colors, alpha=0.6, s=50)
            ax4.plot(days, results['portfolio_values'], alpha=0.3, color='gray', linewidth=1)
            ax4.set_title(f'Trade Timeline ({results["total_trades"]} trades)')
            ax4.set_xlabel('Days')
            ax4.set_ylabel('Portfolio Value at Trade ($)')
            ax4.grid(True, alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='green', label='Buy Actions'),
                             Patch(facecolor='red', label='Sell Actions')]
            ax4.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Backtest visualization saved to: {save_path}")
        plt.close()  # Close to prevent hanging
    
    def compare_with_benchmark(self, results, benchmark_symbol='SPY'):
        """Compare portfolio performance with benchmark"""
        print(f"\n📊 Benchmark Comparison ({benchmark_symbol})")
        print("-" * 40)
        
        try:
            # Fetch benchmark data for same period
            start_date = datetime.now() - timedelta(days=len(results['portfolio_values']))
            end_date = datetime.now()
            benchmark_data = yf.download(benchmark_symbol, start=start_date, end=end_date, progress=False)
            
            if len(benchmark_data) > 0:
                benchmark_start = benchmark_data['Close'].iloc[0]
                benchmark_end = benchmark_data['Close'].iloc[-1]
                benchmark_return = (benchmark_end - benchmark_start) / benchmark_start * 100
                
                print(f"Portfolio Return:    {results['total_return_pct']:+7.2f}%")
                print(f"{benchmark_symbol} Return:         {benchmark_return:+7.2f}%")
                print(f"Outperformance:      {results['total_return_pct'] - benchmark_return:+7.2f}%")
                
                if results['total_return_pct'] > benchmark_return:
                    print("🎉 Portfolio OUTPERFORMED benchmark!")
                else:
                    print("📉 Portfolio underperformed benchmark")
            else:
                print(f"❌ Could not fetch benchmark data for {benchmark_symbol}")
                
        except Exception as e:
            print(f"❌ Benchmark comparison failed: {e}")


class PortfolioTestEnvironment:
    """Simplified test environment (no training, just execution)"""
    
    def __init__(self, stock_data_dict, initial_balance, max_loss_pct=0.30, profit_target_pct=0.50):
        self.stock_data_dict = stock_data_dict
        self.symbols = list(stock_data_dict.keys())
        self.initial_balance = initial_balance
        self.max_loss_pct = max_loss_pct
        self.profit_target_pct = profit_target_pct
        self.max_position_pct = 0.25
        
        # Risk thresholds
        self.stop_loss_threshold = initial_balance * (1 - max_loss_pct)
        self.profit_target = initial_balance * (1 + profit_target_pct)
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = {symbol: 0 for symbol in self.symbols}
        self.total_trades = 0
        return self._get_observation()
    
    def _get_observation(self):
        """Simplified observation for testing"""
        if self.current_step >= len(list(self.stock_data_dict.values())[0]):
            return np.zeros(4 + len(self.symbols) * 8)
        
        # Portfolio features
        portfolio_value = self._calculate_portfolio_value()
        balance_ratio = self.balance / self.initial_balance
        portfolio_return = (portfolio_value - self.initial_balance) / self.initial_balance
        
        # Diversification
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
        
        # Stock features (simplified)
        stock_features = []
        for symbol in self.symbols:
            if self.current_step < len(self.stock_data_dict[symbol]):
                price = float(self.stock_data_dict[symbol].iloc[self.current_step]['Close'])
                volume = float(self.stock_data_dict[symbol].iloc[self.current_step]['Volume']) / 1000000
                holdings = self.shares_held[symbol]
                position_value = holdings * price
                
                # Simple features
                features = [
                    price / 100, volume, price / 100, price / 100, 0.5,  # Simplified tech indicators
                    holdings / 100, position_value / self.initial_balance, 0  # Position info
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
        
        # Execute simplified actions
        if action_type == 1:  # Buy
            max_investment = min(self.balance * 0.8, portfolio_value * self.max_position_pct)
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
    
    best_files = [f for f in os.listdir(models_dir) if f.startswith('best_portfolio_model') and f.endswith('.h5')]
    
    if not best_files:
        # Look for any model files
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
        if model_files:
            print("⚠️ No 'best' model found, using most recent model...")
            model_files.sort(reverse=True)
            model_file = model_files[0]
        else:
            print("❌ No model files found!")
            return None, None
    else:
        # Use the most recent best model
        best_files.sort(reverse=True)
        model_file = best_files[0]
    
    model_path = os.path.join(models_dir, model_file)
    metadata_path = model_path.replace('.h5', '_metadata.pkl')
    
    if not os.path.exists(metadata_path):
        print(f"⚠️ Metadata file not found: {metadata_path}")
        print("🔄 Looking for any metadata file...")
        metadata_files = [f for f in os.listdir(models_dir) if f.endswith('_metadata.pkl')]
        if metadata_files:
            metadata_files.sort(reverse=True)
            metadata_path = os.path.join(models_dir, metadata_files[0])
            print(f"✅ Using: {metadata_files[0]}")
        else:
            print("❌ No metadata files found!")
            return None, None
    
    print(f"✅ Found model: {model_file}")
    print(f"✅ Found metadata: {os.path.basename(metadata_path)}")
    
    return model_path, metadata_path


# Testing script
if __name__ == "__main__":
    print("🧪 Portfolio Model Testing System")
    print("=" * 50)
    
    # Find the best model automatically
    model_path, metadata_path = find_best_model()
    
    if model_path and metadata_path:
        try:
            # Create tester
            tester = PortfolioModelTester(model_path, metadata_path)
            
            # Run backtest on recent data
            print(f"\n🚀 Running backtest on last 90 days of data...")
            results = tester.run_backtest(days_back=90, verbose=True)
            
            # Create visualization
            tester.plot_backtest_results(results, "portfolio_backtest_results.png")
            
            # Compare with S&P 500
            tester.compare_with_benchmark(results, 'SPY')
            
            print(f"\n✅ Testing completed!")
            print(f"📊 Results visualization saved as 'portfolio_backtest_results.png'")
            
        except Exception as e:
            print(f"❌ Testing failed: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print("❌ Could not find trained model files.")
        print("💡 Make sure you have run the training script first!")
        print("📁 Expected files in 'portfolio_models/' directory:")
        print("   • best_portfolio_model_*.h5")
        print("   • best_portfolio_model_*_metadata.pkl")