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

def calculate_bollinger_bands(data, window, num_std_dev):
    """Calculate Bollinger Bands."""
    sma = data['close'].rolling(window=window).mean()
    rolling_std = data['close'].rolling(window=window).std()
    
    upper_band = sma + (rolling_std * num_std_dev)
    lower_band = sma - (rolling_std * num_std_dev)
    
    return sma, upper_band, lower_band

def generate_bollinger_signals(data, window, num_std_dev):
    """Generate buy/sell signals based on Bollinger Bands."""
    sma, upper_band, lower_band = calculate_bollinger_bands(data, window, num_std_dev)
    signals = np.zeros(len(data))
    
    for i in range(len(data)):
        # Buy signal: price crosses below lower band
        if data['close'].iloc[i] < lower_band.iloc[i]:
            signals[i] = 1
        # Sell signal: price crosses above upper band
        elif data['close'].iloc[i] > upper_band.iloc[i]:
            signals[i] = -1
            
    return signals

def backtesting_bollinger(data, window, num_std_dev, initial_balance=100000):
    """Backtest a Bollinger Bands trading strategy."""
    balance = initial_balance
    stock_owned = 0
    
    # Calculate the signals once
    signals = generate_bollinger_signals(data, window, num_std_dev)
    
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

# تست استراتژی بولینگر باند
results = []

# حلقه برای تست ترکیب‌های مختلف پارامترها
for window in range(50, 150):  # طول میانگین متحرک
    for num_std_dev in range(4, 7):  # تعداد انحراف معیار
        final_balance, profit_percentage = backtesting_bollinger(data, window, num_std_dev, initial_balance=100000)
        # ذخیره نتایج در لیست
        results.append({
            'final_balance': final_balance,
            'profit_percentage': profit_percentage,
            'window': window,
            'num_std_dev': num_std_dev
        })
        print(f"Final Balance: {final_balance} , Profit Percentage: {profit_percentage:.2f}% , Window: {window}, Std Dev: {num_std_dev}")

# تبدیل نتایج به DataFrame و ذخیره در فایل CSV
df = pd.DataFrame(results)
df.to_csv('output/bollinger_bands_results3.csv', index=False)
