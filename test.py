# Add this to your debug script to analyze the issues deeper

class ModelDiagnostics:
    """Diagnose specific model issues"""
    
    def __init__(self, tester):
        self.tester = tester
        self.model = tester.model
        self.symbols = tester.symbols
    
    def analyze_action_distribution(self, days_back=30):
        """Analyze why model prefers certain actions"""
        print(f"\n🔍 DEEP ACTION ANALYSIS")
        print("=" * 50)
        
        test_data = self.tester.fetch_test_data(days_back=days_back)
        env = DebugPortfolioEnvironment(test_data, self.tester.initial_balance)
        
        state = env.reset()
        
        # Analyze Q-values for all actions
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        
        print(f"📊 Q-Values Analysis (First State):")
        action_analysis = []
        
        for action_idx in range(len(q_values)):
            stock_idx = action_idx // 5
            action_type = action_idx % 5
            if stock_idx < len(self.symbols):
                symbol = self.symbols[stock_idx]
                action_names = ['Hold', 'Buy', 'Sell', 'Buy More', 'Sell Half']
                action_name = action_names[action_type]
                q_value = q_values[action_idx]
                
                action_analysis.append({
                    'action_idx': action_idx,
                    'symbol': symbol,
                    'action': action_name,
                    'q_value': q_value,
                    'stock_idx': stock_idx,
                    'action_type': action_type
                })
        
        # Sort by Q-value
        action_analysis.sort(key=lambda x: x['q_value'], reverse=True)
        
        print(f"🏆 Top 15 Q-Values:")
        for i, action in enumerate(action_analysis[:15]):
            print(f"   {i+1:2d}. {action['symbol']:4s} {action['action']:9s} = {action['q_value']:7.3f}")
        
        print(f"\n📉 Bottom 10 Q-Values:")
        for i, action in enumerate(action_analysis[-10:]):
            rank = len(action_analysis) - 10 + i + 1
            print(f"   {rank:2d}. {action['symbol']:4s} {action['action']:9s} = {action['q_value']:7.3f}")
        
        # Analyze by stock
        print(f"\n📊 Q-Values by Stock:")
        for symbol in self.symbols:
            stock_actions = [a for a in action_analysis if a['symbol'] == symbol]
            if stock_actions:
                avg_q = sum(a['q_value'] for a in stock_actions) / len(stock_actions)
                max_q = max(a['q_value'] for a in stock_actions)
                min_q = min(a['q_value'] for a in stock_actions)
                print(f"   {symbol:4s}: Avg={avg_q:6.3f}, Max={max_q:6.3f}, Min={min_q:6.3f}")
        
        # Analyze by action type
        print(f"\n📊 Q-Values by Action Type:")
        action_names = ['Hold', 'Buy', 'Sell', 'Buy More', 'Sell Half']
        for action_type, action_name in enumerate(action_names):
            type_actions = [a for a in action_analysis if a['action_type'] == action_type]
            if type_actions:
                avg_q = sum(a['q_value'] for a in type_actions) / len(type_actions)
                count = len(type_actions)
                print(f"   {action_name:9s}: Avg={avg_q:6.3f} ({count} actions)")
        
        return action_analysis
    
    def analyze_observation_space(self, days_back=30):
        """Check if observation space is balanced"""
        print(f"\n🔍 OBSERVATION SPACE ANALYSIS")
        print("=" * 50)
        
        test_data = self.tester.fetch_test_data(days_back=days_back)
        env = DebugPortfolioEnvironment(test_data, self.tester.initial_balance)
        
        state = env.reset()
        print(f"📏 Observation Shape: {state.shape}")
        
        # Analyze portfolio features (first 4 elements)
        portfolio_features = state[:4]
        print(f"🏦 Portfolio Features:")
        feature_names = ['Balance Ratio', 'Return', 'Diversification', 'Risk Exposure']
        for i, (name, value) in enumerate(zip(feature_names, portfolio_features)):
            print(f"   {i}: {name:15s} = {value:.4f}")
        
        # Analyze stock features (8 per stock)
        print(f"\n📈 Stock Features (8 per stock):")
        stock_features = state[4:]
        features_per_stock = 8
        
        for i, symbol in enumerate(self.symbols):
            start_idx = i * features_per_stock
            end_idx = start_idx + features_per_stock
            stock_obs = stock_features[start_idx:end_idx]
            
            print(f"   {symbol:4s}: {stock_obs}")
            
            # Check for problematic values
            if np.all(stock_obs == 0):
                print(f"        ⚠️ ALL ZEROS - Data issue!")
            elif np.any(np.isnan(stock_obs)):
                print(f"        ⚠️ NaN VALUES - Data issue!")
            elif np.any(np.isinf(stock_obs)):
                print(f"        ⚠️ INF VALUES - Scaling issue!")
    
    def test_action_execution(self):
        """Test if actions are executed correctly"""
        print(f"\n🔍 ACTION EXECUTION TEST")
        print("=" * 50)
        
        test_data = self.tester.fetch_test_data(days_back=10)
        env = AggressivePortfolioEnvironment(test_data, self.tester.initial_balance, 0.0)
        
        state = env.reset()
        initial_balance = env.balance
        
        print(f"💰 Initial State:")
        print(f"   Balance: ${env.balance:,.0f}")
        print(f"   Holdings: {env.shares_held}")
        
        # Test buy actions for each stock
        for i, symbol in enumerate(self.symbols):
            buy_action = i * 5 + 1  # Buy action for stock i
            print(f"\n🛒 Testing BUY {symbol} (action {buy_action}):")
            
            # Reset environment
            env.reset()
            state = env._get_observation()
            
            # Execute buy
            next_state, reward, done, info = env.step(buy_action)
            
            print(f"   Before: Balance=${initial_balance:,.0f}")
            print(f"   After:  Balance=${info['balance']:,.0f}")
            print(f"   Shares: {env.shares_held[symbol]}")
            print(f"   Trade executed: {info['balance'] < initial_balance}")
    
    def suggest_fixes(self):
        """Suggest specific fixes based on analysis"""
        print(f"\n💡 SPECIFIC FIXES NEEDED")
        print("=" * 60)
        
        print(f"🎯 1. REWARD STRUCTURE CHANGES:")
        print(f"   • Increase trading rewards (currently too conservative)")
        print(f"   • Add diversification bonus for holding multiple stocks")
        print(f"   • Reduce penalties for reasonable losses")
        print(f"   • Add opportunity cost penalty for holding too much cash")
        
        print(f"\n🔧 2. ACTION SPACE FIXES:")
        print(f"   • Check action encoding/decoding logic")
        print(f"   • Ensure all stocks have equal action representation")
        print(f"   • Consider simplifying to 3 actions: Buy/Hold/Sell")
        
        print(f"\n📊 3. OBSERVATION IMPROVEMENTS:")
        print(f"   • Add relative performance features (stock vs market)")
        print(f"   • Include momentum indicators")
        print(f"   • Add volatility measures")
        print(f"   • Normalize all features properly")
        
        print(f"\n🏃 4. TRAINING CHANGES:")
        print(f"   • Increase exploration (higher epsilon)")
        print(f"   • Use experience replay with diverse scenarios")
        print(f"   • Train on different market conditions")
        print(f"   • Longer training episodes")
        
        print(f"\n⚖️ 5. RISK MANAGEMENT:")
        print(f"   • Set minimum investment thresholds (e.g., 60% invested)")
        print(f"   • Add position sizing rules")
        print(f"   • Implement proper portfolio rebalancing")


# Add to your main script:
def enhanced_diagnostics():
    """Run enhanced diagnostics"""
    model_path, metadata_path = find_best_model()
    
    if model_path and metadata_path:
        try:
            tester = EnhancedPortfolioTester(model_path, metadata_path)
            diagnostics = ModelDiagnostics(tester)
            
            # Run deep analysis
            print("\n" + "="*80)
            print("🔬 DEEP MODEL DIAGNOSTICS")
            print("="*80)
            
            # 1. Action distribution analysis
            diagnostics.analyze_action_distribution()
            
            # 2. Observation space analysis
            diagnostics.analyze_observation_space()
            
            # 3. Action execution test
            diagnostics.test_action_execution()
            
            # 4. Suggest fixes
            diagnostics.suggest_fixes()
            
        except Exception as e:
            print(f"❌ Diagnostics failed: {e}")
            import traceback
            traceback.print_exc()


# Quick fix suggestions for immediate improvement:
def quick_fix_suggestions():
    """Immediate actionable fixes"""
    print(f"\n🚀 IMMEDIATE ACTIONS TO TAKE")
    print("="*50)
    
    print(f"1. 📝 Modify your training script:")
    print(f"   - Increase minimum investment requirement (60-80%)")
    print(f"   - Add diversification rewards")
    print(f"   - Reduce cash-holding rewards")
    
    print(f"\n2. 🔧 Check your action encoding:")
    print(f"   - Verify action_to_stock_action() function")
    print(f"   - Ensure balanced action representation")
    
    print(f"\n3. 📊 Improve observations:")
    print(f"   - Add market-relative features")
    print(f"   - Include technical indicators")
    
    print(f"\n4. 🎯 Retrain with:")
    print(f"   - Higher exploration rate")
    print(f"   - Diverse market conditions")
    print(f"   - Longer episodes (500+ steps)")

if __name__ == "__main__":
    # Run the enhanced diagnostics
    enhanced_diagnostics()
    quick_fix_suggestions()