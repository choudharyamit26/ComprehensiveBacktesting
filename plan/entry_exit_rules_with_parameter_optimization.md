# Detailed Entry/Exit Rules and Parameter Optimization for Trading Strategies

## TIER 1 STRATEGIES - SINGLE INDICATOR SYSTEMS

### Strategy 1: RSI Oversold/Overbought Reversal

#### Entry Rules:
**Long Entry:**
1. RSI(14) < Oversold_Level (default 30)
2. Price must be within 2% of daily low
3. Volume > 1.2x 20-period average volume
4. Time filter: Not in last 30 minutes of trading
5. Wait for RSI to turn up (RSI[0] > RSI[1])

**Short Entry:**
1. RSI(14) > Overbought_Level (default 70)
2. Price must be within 2% of daily high
3. Volume > 1.2x 20-period average volume
4. Time filter: Not in first 30 minutes of trading
5. Wait for RSI to turn down (RSI[0] < RSI[1])

#### Exit Rules:
**Long Exit:**
1. **Profit Target**: RSI reaches 50 OR price gains 1.5% OR R:R ratio 2:1
2. **Stop Loss**: Price falls below entry - (1.5 x ATR) OR RSI makes new low
3. **Time Stop**: Close position after 2 hours if no clear direction
4. **Trailing Stop**: Once 1% profit, trail stop at breakeven

**Short Exit:**
1. **Profit Target**: RSI reaches 50 OR price drops 1.5% OR R:R ratio 2:1
2. **Stop Loss**: Price rises above entry + (1.5 x ATR) OR RSI makes new high
3. **Time Stop**: Close position after 2 hours if no clear direction
4. **Trailing Stop**: Once 1% profit, trail stop at breakeven

#### Parameter Optimization:
```
RSI_Period: [8, 10, 14, 18, 21, 25]
Oversold_Level: [15, 20, 25, 30, 35]
Overbought_Level: [65, 70, 75, 80, 85]
Volume_Multiplier: [1.0, 1.2, 1.5, 2.0]
ATR_Multiplier: [1.0, 1.5, 2.0, 2.5]
Time_Stop_Hours: [1, 1.5, 2, 3, 4]
Profit_Target_%: [1.0, 1.5, 2.0, 2.5]
```

**Optimization Objective Function:**
```
Fitness = (Total_Return * Win_Rate) / Max_Drawdown
- Minimum trades per year: 50
- Maximum drawdown constraint: 15%
- Minimum win rate: 45%
```

---

### Strategy 2: MACD Signal Line Crossover

#### Entry Rules:
**Long Entry:**
1. MACD Line crosses above Signal Line (bullish crossover)
2. MACD Histogram turns positive (MACD_Hist[0] > 0)
3. Both MACD and Signal line below zero (oversold condition)
4. Price above VWAP OR above 9-period EMA
5. Volume surge: Current volume > 1.5x average volume
6. ADX > 20 (trending market filter)

**Short Entry:**
1. MACD Line crosses below Signal Line (bearish crossover)
2. MACD Histogram turns negative (MACD_Hist[0] < 0)
3. Both MACD and Signal line above zero (overbought condition)
4. Price below VWAP OR below 9-period EMA
5. Volume surge: Current volume > 1.5x average volume
6. ADX > 20 (trending market filter)

#### Exit Rules:
**Long Exit:**
1. **Signal Reversal**: MACD crosses below Signal line
2. **Histogram Warning**: MACD Histogram decreasing for 3 consecutive bars
3. **Profit Target**: 2% gain OR 3:1 Risk-Reward ratio
4. **Stop Loss**: 1% loss OR price below recent swing low
5. **Time Decay**: Close after 4 hours if flat

**Short Exit:**
1. **Signal Reversal**: MACD crosses above Signal line
2. **Histogram Warning**: MACD Histogram increasing for 3 consecutive bars
3. **Profit Target**: 2% drop OR 3:1 Risk-Reward ratio
4. **Stop Loss**: 1% loss OR price above recent swing high
5. **Time Decay**: Close after 4 hours if flat

#### Parameter Optimization:
```
MACD_Fast_EMA: [8, 10, 12, 15]
MACD_Slow_EMA: [21, 24, 26, 30]  
MACD_Signal: [7, 9, 12, 15]
Volume_Threshold: [1.2, 1.5, 2.0, 2.5]
ADX_Minimum: [15, 20, 25, 30]
Profit_Target: [1.5, 2.0, 2.5, 3.0]
Stop_Loss_%: [0.8, 1.0, 1.2, 1.5]
Time_Exit_Hours: [2, 3, 4, 6]
```

---

### Strategy 3: Bollinger Band Squeeze Breakout

#### Entry Rules:
**Long Entry:**
1. BB Width at 20-day minimum (squeeze condition)
2. Price breaks above Upper BB with conviction
3. Volume > 2.0x average volume on breakout
4. RSI > 50 (momentum confirmation)
5. Price closes above Upper BB for entry confirmation
6. No major resistance within 2% above entry

**Short Entry:**
1. BB Width at 20-day minimum (squeeze condition)
2. Price breaks below Lower BB with conviction
3. Volume > 2.0x average volume on breakdown
4. RSI < 50 (momentum confirmation)
5. Price closes below Lower BB for entry confirmation
6. No major support within 2% below entry

#### Exit Rules:
**Long Exit:**
1. **Expansion Target**: Price reaches 2x BB Width from middle band
2. **Momentum Loss**: RSI falls below 40
3. **Volume Exhaustion**: Volume drops below 0.8x average for 2 bars
4. **Stop Loss**: Price closes below Middle BB
5. **Trailing Stop**: Trail at Lower BB once profit > 1%

**Short Exit:**
1. **Expansion Target**: Price reaches 2x BB Width from middle band  
2. **Momentum Loss**: RSI rises above 60
3. **Volume Exhaustion**: Volume drops below 0.8x average for 2 bars
4. **Stop Loss**: Price closes above Middle BB
5. **Trailing Stop**: Trail at Upper BB once profit > 1%

#### Parameter Optimization:
```
BB_Period: [15, 20, 25, 30]
BB_StdDev: [1.8, 2.0, 2.2, 2.5]
Volume_Breakout_Multiplier: [1.8, 2.0, 2.5, 3.0]
RSI_Confirmation: [45, 50, 55]
Squeeze_Lookback: [15, 20, 25, 30]
Target_Multiplier: [1.5, 2.0, 2.5, 3.0]
Trail_Stop_Profit_Threshold: [0.8, 1.0, 1.2, 1.5]
```

---

## TIER 2 STRATEGIES - DUAL INDICATOR COMBINATIONS

### Strategy 24: RSI + MACD Momentum Confluence

#### Entry Rules:
**Long Entry:**
1. **RSI Condition**: RSI crosses above 50 AND RSI > RSI[1]
2. **MACD Condition**: MACD line > Signal line AND MACD Histogram > 0
3. **Momentum Alignment**: Both indicators showing upward momentum
4. **Volume Confirmation**: Volume > 1.3x 20-period average
5. **Trend Filter**: Price > 21-period EMA
6. **Time Filter**: Between 10:00 AM - 3:00 PM EST

**Short Entry:**
1. **RSI Condition**: RSI crosses below 50 AND RSI < RSI[1]
2. **MACD Condition**: MACD line < Signal line AND MACD Histogram < 0
3. **Momentum Alignment**: Both indicators showing downward momentum
4. **Volume Confirmation**: Volume > 1.3x 20-period average
5. **Trend Filter**: Price < 21-period EMA
6. **Time Filter**: Between 10:00 AM - 3:00 PM EST

#### Exit Rules:
**Long Exit Priority System:**
1. **Primary Exit**: RSI > 70 AND MACD Histogram decreasing
2. **Momentum Failure**: RSI crosses below 50 OR MACD bearish cross
3. **Profit Target**: 2.5% gain OR when RSI reaches 75
4. **Stop Loss**: 1.2% loss OR price below 21-EMA by 0.5%
5. **Time Stop**: 3 hours maximum hold time
6. **Trailing**: Once 1.5% profit, trail stop at 21-EMA

**Short Exit Priority System:**
1. **Primary Exit**: RSI < 30 AND MACD Histogram increasing
2. **Momentum Failure**: RSI crosses above 50 OR MACD bullish cross
3. **Profit Target**: 2.5% drop OR when RSI reaches 25
4. **Stop Loss**: 1.2% loss OR price above 21-EMA by 0.5%
5. **Time Stop**: 3 hours maximum hold time
6. **Trailing**: Once 1.5% profit, trail stop at 21-EMA

#### Advanced Parameter Optimization:
```python
# Multi-dimensional parameter space
optimization_space = {
    'RSI_Period': [10, 12, 14, 16, 18, 21],
    'RSI_Entry_Level': [45, 50, 55],
    'RSI_Exit_Level': [70, 75, 80, 85],
    'MACD_Fast': [8, 10, 12, 15],
    'MACD_Slow': [21, 24, 26, 30],
    'MACD_Signal': [7, 9, 12],
    'EMA_Period': [20, 21, 26, 30],
    'Volume_Multiplier': [1.2, 1.3, 1.5, 2.0],
    'Profit_Target': [2.0, 2.5, 3.0, 3.5],
    'Stop_Loss': [1.0, 1.2, 1.5, 2.0],
    'Time_Stop_Hours': [2, 3, 4, 5],
    'Trail_Trigger': [1.0, 1.5, 2.0]
}

# Fitness function with multiple objectives
def fitness_function(returns, trades, drawdown, win_rate):
    if trades < 30:  # Minimum trade frequency
        return 0
    
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    calmar_ratio = returns.sum() / abs(drawdown) if drawdown != 0 else 0
    profit_factor = wins_sum / losses_sum if losses_sum != 0 else wins_sum
    
    # Weighted multi-objective score
    score = (0.4 * sharpe_ratio + 
             0.3 * calmar_ratio + 
             0.2 * profit_factor + 
             0.1 * win_rate)
    
    # Penalty for high drawdown
    if drawdown > 0.15:
        score *= 0.5
        
    return score
```

---

### Strategy 30: EMA Crossover with Volume Confirmation

#### Entry Rules:
**Long Entry Sequence:**
1. **Pre-setup**: 9-EMA approaching 21-EMA from below
2. **Crossover**: 9-EMA crosses above 21-EMA
3. **Price Confirmation**: Price closes above both EMAs
4. **Volume Spike**: Volume > 1.8x 20-period average on crossover bar
5. **Momentum Filter**: ADX > 20 and rising
6. **Market Structure**: Price breaks recent swing high
7. **Risk Assessment**: ATR not exceeding 150% of 20-day average

**Short Entry Sequence:**
1. **Pre-setup**: 9-EMA approaching 21-EMA from above
2. **Crossover**: 9-EMA crosses below 21-EMA
3. **Price Confirmation**: Price closes below both EMAs
4. **Volume Spike**: Volume > 1.8x 20-period average on crossover bar
5. **Momentum Filter**: ADX > 20 and rising
6. **Market Structure**: Price breaks recent swing low
7. **Risk Assessment**: ATR not exceeding 150% of 20-day average

#### Dynamic Exit System:
**Long Exit Logic Tree:**
```
IF (Profit > 2% AND Volume < 0.7x Average) THEN
    EXIT with "Volume Exhaustion"
ELIF (9-EMA crosses below 21-EMA) THEN
    EXIT with "Trend Reversal"
ELIF (Price < 9-EMA for 3 consecutive bars) THEN
    EXIT with "Support Break"
ELIF (ADX < 15) THEN
    EXIT with "Trend Weakness"
ELIF (Time_Held > 4 hours AND Profit < 0.5%) THEN
    EXIT with "Time Decay"
ELIF (Loss > 1.5%) THEN
    EXIT with "Stop Loss"
ELSE
    CONTINUE holding with trailing stop
```

#### Position Sizing & Risk Management:
```python
def calculate_position_size(account_equity, atr, entry_price):
    """
    Dynamic position sizing based on volatility
    """
    risk_per_trade = account_equity * 0.02  # 2% risk
    stop_distance = atr * 1.5  # Stop loss distance
    shares = risk_per_trade / stop_distance
    
    # Maximum position constraints
    max_position_value = account_equity * 0.1  # 10% max per position
    max_shares = max_position_value / entry_price
    
    return min(shares, max_shares)

def dynamic_stop_loss(entry_price, current_price, atr, profit_percent):
    """
    Adaptive stop loss based on profit level
    """
    if profit_percent > 3.0:
        return current_price - (atr * 0.8)  # Tight trail
    elif profit_percent > 1.5:
        return current_price - (atr * 1.2)  # Medium trail
    elif profit_percent > 0:
        return entry_price  # Breakeven
    else:
        return entry_price - (atr * 1.5)  # Initial stop
```

---

## TIER 3 STRATEGIES - TRIPLE INDICATOR SYSTEMS

### Strategy 48: RSI + MACD + Volume Triple Confirmation

#### Entry Rules - Multi-Stage Validation:
**Stage 1 - Market Environment Check:**
1. Market not in extreme VIX condition (VIX < 35)
2. Sector ETF showing same directional bias
3. No major news events in next 2 hours
4. Time between 9:45 AM - 3:30 PM EST

**Stage 2 - Technical Setup:**
**Long Setup:**
1. **RSI Component**: RSI(14) crosses above 50 with momentum (RSI[0] - RSI[2] > 3)
2. **MACD Component**: MACD line > Signal line AND Histogram increasing for 2 bars
3. **Volume Component**: Volume > 1.5x average AND Volume trend increasing
4. **Price Action**: Price making higher highs and higher lows pattern
5. **Support Level**: Price above key support (pivot point or previous resistance)

**Stage 3 - Entry Trigger:**
1. All three indicators align within 5-minute window
2. Price breaks above short-term resistance with volume
3. Bid-ask spread normal (< 0.15% of price)
4. No unusual options activity suggesting insider knowledge

#### Exit Rules - Hierarchical System:
**Priority Level 1 (Immediate Exit):**
1. **Risk Management**: Stop loss hit (1.8% or 2x ATR, whichever is smaller)
2. **Market Event**: Breaking news that affects stock/sector
3. **Technical Breakdown**: Price gaps down through support

**Priority Level 2 (Signal Deterioration):**
1. **RSI Failure**: RSI crosses below 45 OR shows bearish divergence
2. **MACD Weakness**: MACD crosses below signal line OR histogram declining for 3 bars
3. **Volume Decline**: Volume drops below 0.8x average for 2 consecutive bars

**Priority Level 3 (Profit Taking):**
1. **Target Achievement**: 3% profit OR RSI > 75
2. **Risk-Reward**: 2.5:1 ratio achieved
3. **Time-Based**: 4 hours elapsed with profit > 1%

**Trailing Stop System:**
```python
def advanced_trailing_stop(entry_price, current_price, high_since_entry, 
                          rsi_current, volume_trend, time_held):
    profit_pct = (current_price - entry_price) / entry_price * 100
    
    if profit_pct > 4.0:
        # Aggressive trail for large profits
        return high_since_entry * 0.96
    elif profit_pct > 2.5:
        # RSI-based trailing
        if rsi_current > 70:
            return high_since_entry * 0.97
        else:
            return high_since_entry * 0.95
    elif profit_pct > 1.0:
        # Volume-based trailing
        if volume_trend == 'declining':
            return high_since_entry * 0.98
        else:
            return entry_price * 1.005  # Slight profit lock
    else:
        return entry_price * 0.982  # Initial stop
```

---

## ADVANCED PARAMETER OPTIMIZATION METHODOLOGIES

### 1. Walk-Forward Analysis Implementation

```python
class WalkForwardOptimizer:
    def __init__(self, strategy, data, optimization_window=126, 
                 test_window=21, step_size=10):
        self.strategy = strategy
        self.data = data
        self.opt_window = optimization_window  # ~6 months
        self.test_window = test_window         # ~1 month
        self.step_size = step_size            # ~2 weeks
    
    def optimize_parameters(self, start_idx, end_idx):
        """Optimize parameters on training window"""
        train_data = self.data[start_idx:end_idx]
        
        # Genetic Algorithm optimization
        ga = GeneticAlgorithm(
            population_size=50,
            generations=100,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
        best_params = ga.optimize(
            objective_function=self.fitness_function,
            parameter_space=self.strategy.param_space,
            data=train_data
        )
        
        return best_params
    
    def fitness_function(self, params, data):
        """Multi-objective fitness function"""
        results = self.strategy.backtest(params, data)
        
        # Metrics calculation
        total_return = results['returns'].sum()
        volatility = results['returns'].std()
        max_drawdown = results['drawdown'].min()
        win_rate = (results['trades'] > 0).mean()
        trade_count = len(results['trades'])
        
        # Sharpe ratio with minimum trade filter
        if trade_count < 10:
            return 0
            
        sharpe = total_return / volatility if volatility > 0 else 0
        
        # Multi-objective score
        score = (0.4 * sharpe + 
                0.3 * (total_return / abs(max_drawdown)) +
                0.2 * win_rate +
                0.1 * min(trade_count / 50, 1.0))  # Trade frequency bonus
        
        return score
    
    def run_walk_forward(self):
        """Execute complete walk-forward analysis"""
        results = []
        
        for i in range(0, len(self.data) - self.opt_window - self.test_window, 
                      self.step_size):
            # Optimization period
            opt_start = i
            opt_end = i + self.opt_window
            
            # Test period
            test_start = opt_end
            test_end = opt_end + self.test_window
            
            # Optimize parameters
            best_params = self.optimize_parameters(opt_start, opt_end)
            
            # Test on out-of-sample data
            test_data = self.data[test_start:test_end]
            test_results = self.strategy.backtest(best_params, test_data)
            
            results.append({
                'optimization_period': (opt_start, opt_end),
                'test_period': (test_start, test_end),
                'parameters': best_params,
                'test_performance': test_results
            })
        
        return results
```

### 2. Multi-Objective Optimization with Constraints

```python
from scipy.optimize import differential_evolution
import numpy as np

class MultiObjectiveOptimizer:
    def __init__(self, strategy):
        self.strategy = strategy
        
    def objective_function(self, params, data, weights):
        """
        Multi-objective function with constraints
        """
        results = self.strategy.backtest(params, data)
        
        # Primary objectives
        total_return = results['total_return']
        max_drawdown = abs(results['max_drawdown'])
        sharpe_ratio = results['sharpe_ratio']
        win_rate = results['win_rate']
        profit_factor = results['profit_factor']
        trade_count = results['trade_count']
        
        # Constraint penalties
        penalty = 0
        
        # Minimum trade frequency constraint
        if trade_count < 30:
            penalty += 1000
            
        # Maximum drawdown constraint  
        if max_drawdown > 0.15:
            penalty += (max_drawdown - 0.15) * 5000
            
        # Minimum win rate constraint
        if win_rate < 0.40:
            penalty += (0.40 - win_rate) * 1000
        
        # Multi-objective score
        score = (weights['return'] * total_return +
                weights['sharpe'] * sharpe_ratio +
                weights['win_rate'] * win_rate +
                weights['profit_factor'] * profit_factor -
                weights['drawdown'] * max_drawdown -
                penalty)
        
        return -score  # Minimize negative score
    
    def optimize(self, data, param_bounds, weights):
        """
        Differential Evolution optimization
        """
        result = differential_evolution(
            func=self.objective_function,
            bounds=param_bounds,
            args=(data, weights),
            maxiter=200,
            popsize=15,
            mutation=(0.5, 1.5),
            recombination=0.7,
            seed=42,
            polish=True
        )
        
        return result.x, -result.fun
```

### 3. Regime-Aware Parameter Optimization

```python
class RegimeAwareOptimizer:
    def __init__(self, strategy):
        self.strategy = strategy
        self.regimes = ['bull', 'bear', 'sideways', 'high_vol', 'low_vol']
    
    def identify_regime(self, data, window=50):
        """
        Identify market regime based on multiple factors
        """
        returns = data['returns']
        vix = data['vix']
        
        # Trend regime
        sma_50 = data['close'].rolling(50).mean()
        trend_score = (data['close'] / sma_50 - 1).iloc[-1]
        
        # Volatility regime  
        vol_score = vix.iloc[-1] / vix.rolling(252).mean().iloc[-1]
        
        # Classify regime
        if trend_score > 0.05 and vol_score < 1.2:
            return 'bull'
        elif trend_score < -0.05 and vol_score < 1.2:
            return 'bear'
        elif abs(trend_score) < 0.05 and vol_score < 1.2:
            return 'sideways'
        elif vol_score > 1.5:
            return 'high_vol'
        else:
            return 'low_vol'
    
    def optimize_by_regime(self, data):
        """
        Optimize parameters for each market regime
        """
        regime_params = {}
        
        for regime in self.regimes:
            # Filter data by regime
            regime_data = self.filter_by_regime(data, regime)
            
            if len(regime_data) < 100:  # Minimum data requirement
                continue
                
            # Optimize for this regime
            optimizer = MultiObjectiveOptimizer(self.strategy)
            best_params, score = optimizer.optimize(
                regime_data, 
                self.strategy.param_bounds,
                weights={'return': 0.3, 'sharpe': 0.3, 'win_rate': 0.2, 
                        'profit_factor': 0.1, 'drawdown': 0.1}
            )
            
            regime_params[regime] = {
                'parameters': best_params,
                'score': score,
                'sample_size': len(regime_data)
            }
        
        return regime_params
    
    def adaptive_trading(self, current_data, regime_params):
        """
        Select parameters based on current market regime
        """
        current_regime = self.identify_regime(current_data)
        
        if current_regime in regime_params:
            return regime_params[current_regime]['parameters']
        else:
            # Fallback to default parameters
            return self.strategy.default_params
```

### 4. Robustness Testing Framework

```python
class RobustnessTestSuite:
    def __init__(self, strategy, optimized_params):
        self.strategy = strategy
        self.base_params = optimized_params
    
    def parameter_sensitivity_test(self, data, sensitivity_range=0.1):
        """
        Test sensitivity to parameter changes
        """
        results = {}
        
        for param_name, base_value in self.base_params.items():
            if isinstance(base_value, (int, float)):
                # Test ±10% parameter variations
                variations = [
                    base_value * (1 - sensitivity_range),
                    base_value,
                    base_value * (1 + sensitivity_range)
                ]
                
                param_results = []
                for variation in variations:
                    test_params = self.base_params.copy()
                    test_params[param_name] = variation
                    
                    performance = self.strategy.backtest(test_params, data)
                    param_results.append({
                        'value': variation,
                        'performance': performance
                    })
                
                results[param_name] = param_results
        
        return results
    
    def monte_carlo_simulation(self, data, n_simulations=1000):
        """
        Monte Carlo simulation with random parameter variations
        """
        results = []
        
        for _ in range(n_simulations):
            # Generate random parameter set within bounds
            random_params = {}
            for param, bounds in self.strategy.param_bounds.items():
                if isinstance(bounds, tuple):
                    random_params[param] = np.random.uniform(bounds[0], bounds[1])
                else:
                    random_params[param] = np.random.choice(bounds)
            
            # Test performance
            performance = self.strategy.backtest(random_params, data)
            results.append({
                'parameters': random_params,
                'performance': performance
            })
        
        return results
    
    def stress_test(self, data):
        """
        Test strategy under stressed market conditions
        """
        stress_scenarios = {
            'market_crash': self.simulate_crash(data),
            'high_volatility': self.simulate_high_vol(data),
            'trending_market': self.simulate_trend(data),
            'choppy_market': self.simulate_chop(data)
        }
        
        results = {}
        for scenario_name, scenario_data in stress_scenarios.items():
            performance = self.strategy.backtest(self.base_params, scenario_data)
            results[scenario_name] = performance
        
        return results
```

General Guidelines by Complexity
Simple strategies (2-4 parameters):

Minimum: 50-100 trials
Recommended: 200-500 trials
Optimal: 500-1000 trials

Medium complexity (5-8 parameters):

Minimum: 100-200 trials
Recommended: 500-1000 trials
Optimal: 1000-2000 trials

Complex strategies (9+ parameters):

Minimum: 500-1000 trials
Recommended: 2000-5000 trials
Optimal: 5000-10000 trials

