import pandas as pd
import numpy as np

def calculate_moving_averages(data, m1_window, m2_window, m3_window, m4_window):
    """Calculate multiple moving averages."""
    m1 = data['close'].rolling(window=m1_window).mean()
    m2 = data['close'].rolling(window=m2_window).mean()
    m3 = data['close'].rolling(window=m3_window).mean()
    m4 = data['close'].rolling(window=m4_window).mean()
    
    return m1, m2, m3, m4

def calculate_adx(data, window=14):
    """Calculate ADX (Average Directional Index)."""
    high = data['high']
    low = data['low']
    close = data['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Average True Range
    atr = true_range.rolling(window=window).mean()
    
    # Directional Movement
    up_move = high.diff()
    down_move = low.diff().abs()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=window).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=window).mean() / atr)
    
    adx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).rolling(window=window).mean()
    
    return adx

def generate_signals(data, m1_window, m2_window, m3_window, m4_window, adx_window, adx_threshold):
    """Generate buy/sell signals based on moving averages and ADX."""
    m1, m2, m3, m4 = calculate_moving_averages(data, m1_window, m2_window, m3_window, m4_window)
    adx = calculate_adx(data, adx_window)
    
    signals = np.zeros(len(data))
    
    for i in range(len(data)):
        # Check conditions for buy signal
        if (
            # i >= max(m1_window, m2_window, m3_window, m4_window) and
            m1.iloc[i] > m2.iloc[i] and
            m2.iloc[i] > m3.iloc[i] and
            data['close'].iloc[i] > m4.iloc[i] and
            adx.iloc[i] > adx_threshold and
            adx.iloc[i] > adx.iloc[i-1]):
            # adx.iloc[i] > adx_threshold):
            signals[i] = 1  # Buy signal
            
        # Check conditions for sell signal
        elif (
            # i <= max(m1_window, m2_window, m3_window, m4_window) and
              m1.iloc[i] < m2.iloc[i] and
              m2.iloc[i] < m3.iloc[i] and
              data['close'].iloc[i] < m4.iloc[i] and
              adx.iloc[i] > adx_threshold and
              adx.iloc[i] < adx.iloc[i-1]):
              # adx.iloc[i] > adx_threshold):

            signals[i] = -1  # Sell signal
        else:    
            signals[i] = 0
    
    return signals

def backtesting_moving_average(data, m1_window, m2_window, m3_window, m4_window, adx_window, adx_threshold, initial_balance=100000, broker_fee_percent=0.1):
    """Backtest a moving average trading strategy."""
    balance = initial_balance
    stock_owned = 0.0  # Allow fractional ownership
    
    # Calculate the signals once
    signals = generate_signals(data, m1_window, m2_window, m3_window, m4_window, adx_window, adx_threshold)
    
    for index in range(len(data)):
        signal = signals[index]
        price = data['close'].iloc[index]
        
        if signal == 1:  # Buy signal
            # Calculate how much can be bought
            amount_to_invest = balance * (1 - broker_fee_percent / 100)  # Deduct broker fee
            shares_to_buy = amount_to_invest / price  # Fractional shares
            stock_owned += shares_to_buy
            balance -= shares_to_buy * price * (1 + broker_fee_percent / 100)  # Deduct balance with fee
            
        elif signal == -1:  # Sell signal
            if stock_owned > 0:  # Check if there are stocks to sell
                balance += stock_owned * price * (1 - broker_fee_percent / 100)  # Add to balance after selling
                stock_owned = 0  # Reset stock owned after selling
    
    final_balance = balance + (stock_owned * data['close'].iloc[-1])  # Sell remaining stocks at last price
    profit_percentage = ((final_balance - initial_balance) / initial_balance) * 100
    return final_balance, profit_percentage

# مثال استفاده
file_path = 'data/hourly_btc_data.csv'
data = pd.read_csv(file_path)

# مثال استفاده
results = []

# حلقه برای تست ترکیب‌های مختلف پارامترها
for m1_window in range(13, 20):  # طول میانگین متحرک m1
    for i in range(m1_window+1, m1_window+10):  # طول میانگین متحرک m2
        for j in range(i+1, i+10):  # طول میانگین متحرک m4
            for adx in range(14, 19):  # طول میانگین متحرک m4
                final_balance, profit_percentage = backtesting_moving_average(data, m1_window, i, j , i-5, adx_window=adx, adx_threshold=adx+3)
                results.append({
                    'final_balance': final_balance,
                    'profit_percentage': profit_percentage,
                    'm1_window': m1_window,
                    'm2_window': i,
                    'm3_window': j,
                    'm4_window': m1_window-5,
                    'adx' : adx,
                    'adx_threshold' : adx+3,
                    
                    
                })
            print(f"Final Balance: {final_balance} , Profit Percentage: {profit_percentage:.2f}% , m1: {m1_window}, m2: {m1_window+i}, m3: {m1_window+j}, m4: {m1_window}")


# تبدیل نتایج به DataFrame و ذخیره در فایل CSV
df = pd.DataFrame(results)
df.to_csv('output/moving_average_results8.csv', index=False)
