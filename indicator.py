import pandas as pd
import numpy as np

# تابع برای تعیین وضعیت (خرید یا فروش)
def determine_state(current_price, previous_price):
    if current_price > previous_price:
        return 1
    else:
        return 0

# تابع برای محاسبه درصد تغییر
def calculate_percent(current_price, previous_price):
    if previous_price != 0:  # جلوگیری از تقسیم بر صفر
        return ((current_price - previous_price) / previous_price) * 100
    return np.nan  # اگر قیمت قبلی صفر باشد

# تابع برای محاسبه رنک
def calculate_rank(data, start_index, streak_counter, current_state):
    for j in range(start_index, start_index + streak_counter):
        if current_state == 1 :
            data.at[j, 'new_rank'] = streak_counter - (j - start_index)
        else:
            data.at[j, 'new_rank'] = -(streak_counter - (j - start_index))

# تابع برای محاسبه RTP
def calculate_rtp(data, start_index, streak_counter, current_state):
    total_percent = 0  # جمع کل درصد
    k = streak_counter + start_index
    for j in range(start_index, start_index + streak_counter):
        k -= 1
        if current_state == 1 :
            total_percent += data.at[j, 'percent']
        else:
            total_percent -= data.at[j, 'percent']
        
        # ذخیره در ستون new_rtp برای هر ردیف
        data.at[j, 'streak_counter'] = streak_counter
        data.at[j, 'start_index'] = start_index


def calculate_sum_percent(data):
    # ایجاد یک ستون جدید برای Sum_percent
    data['Sum_percent'] = np.nan  # ابتدا مقداردهی به NaN

    # گروه‌بندی بر اساس state و start_index
    grouped = data.groupby(['state', 'start_index'])

    # محاسبه جمع کل برای هر گروه
    for (state, start_index), group in grouped:
        total_sum = group['percent'].sum()  # جمع درصدها برای گروه خاص
        # محاسبه Sum_percent برای هر ردیف در گروه
        data.loc[group.index, 'Sum_percent'] = total_sum - group['percent'].cumsum()
        
# 3. محاسبه وضعیت، درصد و new_rank
def calculate_state_and_new_rank(data):
    current_state = None
    streak_counter = 0  # شمارنده برای تعداد روزهای صعودی یا نزولی

    for i in range(len(data)):
        if i == 0:
            continue  # اولین روز را نادیده می‌گیریم

        # تعیین وضعیت
        state = determine_state(data['close'].iloc[i], data['close'].iloc[i - 1])
        data.at[i, 'state'] = state  # ذخیره وضعیت در DataFrame

        # محاسبه درصد تغییر
        data.at[i, 'percent'] = calculate_percent(data['close'].iloc[i], data['close'].iloc[i - 1])

        # محاسبه رنک جدید
        if state == current_state:
            streak_counter += 1  # افزایش شمارنده روزهای صعودی یا نزولی
        else:
            # اگر وضعیت تغییر کرده است، رنک را تنظیم می‌کنیم
            if current_state is not None:
                calculate_rank(data, i - streak_counter, streak_counter, current_state)

            # بازنشانی شمارنده و تعیین وضعیت جدید
            current_state = state
            streak_counter = 1  # شروع شمارنده برای حالت جدید

    # تنظیم رنک برای آخرین روز
    if current_state is not None:
        calculate_rank(data, len(data) - streak_counter, streak_counter, current_state)

# 6. محاسبه new_rtp
def calculate_new_rtp(data):
    current_state = None
    streak_counter = 0  # شمارنده برای تعداد روزهای صعودی یا نزولی

    for i in range(len(data)):
        if i == 0:
            continue  # اولین روز را نادیده می‌گیریم

        # محاسبه رنک جدید
        if data.at[i, 'state'] == current_state:
            streak_counter += 1  # افزایش شمارنده روزهای صعودی یا نزولی
        else:
            # اگر وضعیت تغییر کرده است، رنک را تنظیم می‌کنیم
            if current_state is not None:
                calculate_rtp(data, i - streak_counter, streak_counter, current_state)

            # بازنشانی شمارنده و تعیین وضعیت جدید
            current_state = data.at[i, 'state']
            streak_counter = 1  # شروع شمارنده برای حالت جدید

    # تنظیم رنک برای آخرین روز
    if current_state is not None:
        calculate_rtp(data, len(data) - streak_counter, streak_counter, current_state)

def calculate_sma(data, window):
    return data['close'].rolling(window=window).mean()

def calculate_rsi(data, window):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# و سایر اندیکاتورهای مورد نظر

def calculate_wma(data, window):
    weights = np.arange(1, window + 1)
    return data['close'].rolling(window=window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

def calculate_volatility(data, window):
    return data['close'].rolling(window=window).std()

def calculate_rsi(data, window):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    ema_short = data['close'].ewm(span=short_window, adjust=False).mean()
    ema_long = data['close'].ewm(span=long_window, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal


def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    sma = data['close'].rolling(window=window).mean()
    std_dev = data['close'].rolling(window=window).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, lower_band


def calculate_pivot_points(data):
    pivot = (data['high'] + data['low'] + data['close']) / 3
    support1 = (2 * pivot) - data['high']
    resistance1 = (2 * pivot) - data['low']
    return pivot, support1, resistance1


def calculate_stochastic(data, k_window=14, d_window=3):
    min_low = data['low'].rolling(window=k_window).min()
    max_high = data['high'].rolling(window=k_window).max()
    stoch_k = 100 * ((data['close'] - min_low) / (max_high - min_low))
    stoch_d = stoch_k.rolling(window=d_window).mean()
    return stoch_k, stoch_d

def calculate_adx(data, window=14):
    high = data['high']
    low = data['low']
    close = data['close']

    # محاسبه تغییرات
    tr = np.maximum(high.diff(), close.shift().diff(), low.diff())
    atr = tr.rolling(window=window).mean()

    # محاسبه +DI و -DI
    plus_dm = np.where((high.diff() > low.diff()) & (high.diff() > 0), high.diff(), 0)
    minus_dm = np.where((low.diff() > high.diff()) & (low.diff() > 0), low.diff(), 0)

    # تبدیل plus_dm و minus_dm به Series
    plus_dm_series = pd.Series(plus_dm, index=data.index)
    minus_dm_series = pd.Series(minus_dm, index=data.index)

    plus_di = 100 * (plus_dm_series.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm_series.rolling(window=window).mean() / atr)

    adx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di)).rolling(window=window).mean()
    return adx

def calculate_cci(data, window=20):
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    sma = typical_price.rolling(window=window).mean()
    mean_deviation = (typical_price - sma).abs().rolling(window=window).mean()
    cci = (typical_price - sma) / (0.015 * mean_deviation)
    return cci


def calculate_classic_pivot(data):
    pivot = (data['high'] + data['low'] + data['close']) / 3
    support1 = (2 * pivot) - data['high']
    resistance1 = (2 * pivot) - data['low']
    support2 = pivot - (data['high'] - data['low'])
    resistance2 = pivot + (data['high'] - data['low'])
    return pivot, support1, resistance1, support2, resistance2


def calculate_atr(data, window=14):
    high_low = data['high'] - data['low']
    high_close = (data['high'] - data['close'].shift()).abs()
    low_close = (data['low'] - data['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def calculate_mfi(data, window=14):
    # بررسی وجود ستون 'volume'
    if 'volume' not in data.columns:
        raise KeyError("Column 'volume' not found in the DataFrame.")

    typical_price = (data['high'] + data['low'] + data['close']) / 3
    money_flow = typical_price * data['volume']
    positive_flow = money_flow.where(data['close'].diff() > 0, 0).rolling(window=window).sum()
    negative_flow = money_flow.where(data['close'].diff() < 0, 0).rolling(window=window).sum()

    # جلوگیری از تقسیم بر صفر
    negative_flow = negative_flow.replace(0, np.nan)  # جلوگیری از تقسیم بر صفر
    mfi = 100 - (100 / (1 + (positive_flow / negative_flow)))
    
    return mfi

def calculate_aroon(data, window=14):
    aroon_up = ((window - data['high'].rolling(window=window).apply(lambda x: x.argmax() + 1)) / window) * 100
    aroon_down = ((window - data['low'].rolling(window=window).apply(lambda x: x.argmin() + 1)) / window) * 100
    return aroon_up, aroon_down

def calculate_williams_r(data, window=14):
    highest_high = data['high'].rolling(window=window).max()
    lowest_low = data['low'].rolling(window=window).min()
    williams_r = -100 * (highest_high - data['close']) / (highest_high - lowest_low)
    return williams_r

def calculate_roc(data, window=12):
    roc = (data['close'].diff(window) / data['close'].shift(window)) * 100
    return roc

def calculate_nvi(data):
    nvi = pd.Series(index=data.index, data=0.0)
    for i in range(1, len(data)):
        if data['volume'].iloc[i] < data['volume'].iloc[i - 1]:
            nvi.iloc[i] = nvi.iloc[i - 1]
        else:
            nvi.iloc[i] = nvi.iloc[i - 1] + (data['close'].iloc[i] - data['close'].iloc[i - 1]) / data['close'].iloc[i - 1] * 100
    return nvi


def calculate_cmf(data, window=20):
    money_flow_multiplier = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
    money_flow_volume = money_flow_multiplier * data['volume']
    cmf = money_flow_volume.rolling(window=window).sum() / data['volume'].rolling(window=window).sum()
    return cmf


def calculate_cmo(data, window=14):
    gain = (data['close'].diff().where(data['close'].diff() > 0, 0)).rolling(window=window).sum()
    loss = (-data['close'].diff().where(data['close'].diff() < 0, 0)).rolling(window=window).sum()
    cmo = ((gain - loss) / (gain + loss)) * 100
    return cmo


def calculate_tsi(data, long_window=25, short_window=13):
    price_change = data['close'].diff()
    ema1 = price_change.ewm(span=short_window).mean()
    ema2 = ema1.ewm(span=long_window).mean()
    tsi = (ema2 / ema1.abs().ewm(span=long_window).mean()) * 100
    return tsi

