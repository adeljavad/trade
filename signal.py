import pandas as pd
import numpy as np

# file_path = 'data/hourly_btc_data.csv'
# data = pd.read_csv(file_path)

# ۱. میانگین متحرک وزنی (WMA)
def calculate_wma(data, window):
    return data['close'].rolling(window=window).mean()

def signal_wma(data, window):
    wma = calculate_wma(data, window)
    signal = np.where(data['close'] > wma, 1, -1)
    return signal

# ۲. شاخص قدرت نسبی (RSI)
def calculate_rsi(data, window):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def signal_rsi(data, window):
    rsi = calculate_rsi(data, window)
    signal = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
    return signal

# ۳. میانگین متحرک همگرایی و واگرایی (MACD)
def calculate_macd(data, fast_window, slow_window, signal_window):
    exp1 = data['close'].ewm(span=fast_window, adjust=False).mean()
    exp2 = data['close'].ewm(span=slow_window, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, macd_signal

def signal_macd(data, fast_window, slow_window, signal_window):
    macd, macd_signal = calculate_macd(data, fast_window, slow_window, signal_window)
    signal = np.where(macd > macd_signal, 1, -1)
    return signal

# ۴. اندیکاتور استوکاستیک
def calculate_stochastic(data, k_window, d_window):
    min_low = data['low'].rolling(window=k_window).min()
    max_high = data['high'].rolling(window=k_window).max()
    stoch_k = 100 * ((data['close'] - min_low) / (max_high - min_low))
    stoch_d = stoch_k.rolling(window=d_window).mean()
    return stoch_k, stoch_d

def signal_stochastic(data, k_window, d_window):
    stoch_k, stoch_d = calculate_stochastic(data, k_window, d_window)
    signal = np.where((stoch_k > stoch_d) & (stoch_k < 20), 1, np.where((stoch_k < stoch_d) & (stoch_k > 80), -1, 0))
    return signal

# ۵. اندیکاتور ADX
def calculate_adx(data, window):
    high = data['high']
    low = data['low']
    close = data['close']
    
    tr = np.maximum(high.diff(), close.shift().diff(), low.diff())
    atr = tr.rolling(window=window).mean()

    plus_dm = pd.Series(np.where((high.diff() > low.diff()) & (high.diff() > 0), high.diff(), 0))
    minus_dm = pd.Series(np.where((low.diff() > high.diff()) & (low.diff() > 0), low.diff(), 0))

    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)

    adx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di)).rolling(window=window).mean()
    return adx

def signal_adx(data, window):
    adx = calculate_adx(data, window)
    signal = np.where(adx > 25, 1, -1)
    return signal

def signal_buy(data):
    buy_signal = []
    for i in range(len(data)):
        if (i >= 14):  # برای جلوگیری از دسترسی به ایندکس‌های منفی
            # بررسی شرایط
            if (data['MA5'][i] > data['MA14'][i]) and (data['MA5'][i-1] <= data['MA14'][i-1]) and (data['close'][i] > data['MA25'][i]):
                buy_signal.append(1)
            else:
                buy_signal.append(-1)
        else:
            buy_signal.append(0)  # برای ایندکس‌های اولیه
    return buy_signal



# محاسبه میانگین‌های متحرک

# نمایش نتیجه
# print(data[['close',  'MA14', 'MA25', 'Signal_Buy']].tail(20))



# output_file_path = 'output/updated_stock_data3.csv'
# data.to_csv(output_file_path, index=True)


# print(data.head())
