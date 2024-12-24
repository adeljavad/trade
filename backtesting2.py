import numpy as np
import pandas as pd

def calculate_rsi(data, window):
    """Calculate the Relative Strength Index (RSI) for a given data series."""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_moving_average(data, window):
    """Calculate the moving average for a given data series."""
    return data['close'].rolling(window=window).mean()

def generate_rsi_signals(rsi, min_rsi, max_rsi):
    """Generate RSI signals based on the movement of RSI."""
    rsi_signal = np.zeros(len(rsi))  # ایجاد آرایه‌ای برای سیگنال‌ها

    for i in range(1, len(rsi)):
        # سیگنال خرید: RSI در حال صعود است و کمتر از min_rsi است
        if rsi[i-1] < rsi[i] and rsi[i] <= min_rsi:
            rsi_signal[i] = 1  # سیگنال خرید
        # سیگنال فروش: RSI در حال نزول است و بیشتر از max_rsi است
        elif rsi[i-1] > rsi[i] and rsi[i] >= max_rsi:
            rsi_signal[i] = -1  # سیگنال فروش
        # وضعیت خنثی: هیچ سیگنالی صادر نمی‌شود
        else:
            rsi_signal[i] = 0  # وضعیت خنثی
            
    return rsi_signal 

def signal_combined(data, rsi_window, ma_window, min_rsi, max_rsi):
    """Generate combined signals based on RSI and Moving Average."""
    # Calculate RSI and Moving Average once
    rsi = calculate_rsi(data, rsi_window)
    ma = calculate_moving_average(data, ma_window)
    
    # Generate RSI signals
    # rsi_signal = generate_rsi_signals(rsi, min_rsi, max_rsi)
    rsi_signal = np.where(rsi < min_rsi, 1, np.where(rsi > max_rsi, -1, 0))
    # Generate Moving Average signals
    ma_signal = np.where(data['close'] > ma, 1, -1)  # Buy if price is above MA, sell if below
    
    # Combine signals
    combined_signal = np.where((rsi_signal == 1) & (ma_signal == 1), 1, 
                                np.where((rsi_signal == -1) & (ma_signal == -1), -1, 0))
    
    return combined_signal

def backtesting(data, signal_function_name, rsi_window, ma_window, min_rsi, max_rsi, initial_balance=100000):
    """Backtest a trading strategy with dynamic signal evaluation."""
    balance = initial_balance
    stock_owned = 0
    
    # Calculate the signals once
    signals = globals()[signal_function_name](data, rsi_window, ma_window, min_rsi, max_rsi)
    
    for index in range(len(data)):
        signal = signals[index]
        price = data['close'].iloc[index]
        
        if signal == 1:  # Buy signal
            if balance >= price:  # Check if there is enough balance to buy
                stock_owned += 1
                balance -= price
                
        elif signal == -1:  # Sell signal
            if stock_owned > 0:  # Check if there are stocks to sell
                stock_owned -= 1
                balance += price
    
    final_balance = balance + (stock_owned * data['close'].iloc[-1])  # Sell remaining stocks at last price
    profit_percentage = ((final_balance - initial_balance) / initial_balance) * 100
    
    return final_balance, profit_percentage

# مثال استفاده
file_path = 'data/hourly_btc_data.csv'
data = pd.read_csv(file_path)

# استفاده از سیگنال ترکیبی
results = []

# حلقه برای تست ترکیب‌های مختلف پارامترها
for i in range(39, 43):  #best 39,40,41
    for j in range(20, 25): #best  23,24
        for min_rsi, max_rsi in [(30, 70), (29, 71), (28, 72)]:  # best 28,72
            final_balance, profit_percentage = backtesting(data, 'signal_combined', i, j, min_rsi, max_rsi)
            # ذخیره نتایج در لیست
            results.append({
                'final_balance': final_balance,
                'percent': profit_percentage,
                'profit_percentage': profit_percentage,
                'RSI': i,
                'mve': j,
                'min_rsi': min_rsi,
                'max_rsi': max_rsi
            })
            print(f"Final Balance: {final_balance} , Profit Percentage: {profit_percentage:.2f}% , RSI: {i:d} , MVE: {j:d}, MIN: {min_rsi} , MAX: {max_rsi}")

# تبدیل نتایج به DataFrame و ذخیره در فایل CSV
df = pd.DataFrame(results)
df.to_csv('output/rsi_mv2.csv', index=False)
