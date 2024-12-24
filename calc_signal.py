import pandas as pd
import numpy as np
import Signal
import indicator
# 1. خواندن داده‌های CSV
# file_path = 'data/stock_data.csv'  # مسیر فایل CSV
file_path = 'data/hourly_btc_data.csv'
data = pd.read_csv(file_path)

# 2. اضافه کردن ستون‌های جدید با نوع داده مناسب
data['state'] = pd.Series(dtype='object')  # تعیین نوع داده به object
data['percent'] = np.nan
data['new_rank'] = np.nan  # اضافه کردن ستون new_rank
data['Sum_percent'] = np.nan  # اضافه کردن ستون Sum_percent

# 4. اجرای تابع محاسبه وضعیت، درصد و new_rank
indicator.calculate_state_and_new_rank(data)

indicator.calculate_new_rtp(data)
# 7. محاسبه Sum_percent
indicator.calculate_sum_percent(data)



# محاسبه باندهای بولینگر
data['Bollinger_Upper'], data['Bollinger_Lower'] = indicator.calculate_bollinger_bands(data)

# # محاسبه نقاط پیوت
data['Pivot'], data['Support1'], data['Resistance1'] = indicator.calculate_pivot_points(data)

# محاسبه استوکاستیک
data['Stoch_K'], data['Stoch_D'] = indicator.calculate_stochastic(data)

# محاسبه ADX
data['ADX_14'] = indicator.calculate_adx(data)

# محاسبه MFI
data['MFI_14'] = indicator.calculate_mfi(data)

# محاسبه SMA و RSI و اضافه کردن به DataFrame
data['SMA_20'] = indicator.calculate_sma(data, 20)  # SMA با دوره 20
data['RSI_14'] = indicator.calculate_rsi(data, 14)  # RSI با دوره 14
# مثال: محاسبه RSI با دوره 14
data['RSI_9'] = indicator.calculate_rsi(data, 9)

data['EMA_20'] = data['close'].ewm(span=20, adjust=False).mean()  # EMA با دوره 20
data['MACD'] = data['close'].ewm(span=12, adjust=False).mean() - data['close'].ewm(span=26, adjust=False).mean()  # MACD

# محاسبه ATR
data['ATR_14'] = indicator.calculate_atr(data)

# مثال: محاسبه نوسان با دوره 20
data['Volatility_20'] = indicator.calculate_volatility(data, 20)
# محاسبه CCI
data['CCI_20'] = indicator.calculate_cci(data)
# محاسبه نقاط پیوت کلاسیک
data['Classic_Pivot'], data['Classic_Support1'], data['Classic_Resistance1'], data['Classic_Support2'], data['Classic_Resistance2'] = indicator.calculate_classic_pivot(data)
# محاسبه Williams %R
data['Williams_R'] = indicator.calculate_williams_r(data)
# محاسبه ROC
data['ROC_12'] = indicator.calculate_roc(data)
# محاسبه NVI
data['NVI'] = indicator.calculate_nvi(data)
# محاسبه Aroon
data['Aroon_Up'], data['Aroon_Down'] = indicator.calculate_aroon(data)
# محاسبه CMF
data['CMF_20'] = indicator.calculate_cmf(data)
# محاسبه CMO
data['CMO_14'] = indicator.calculate_cmo(data)
# محاسبه TSI
data['TSI'] = indicator.calculate_tsi(data)
# مثال: محاسبه WMA با دوره 20
data['WMA_20'] = indicator.calculate_wma(data, 20)

data['MA5'] = data['close'].rolling(window=3).mean()
data['MA14'] = data['close'].rolling(window=10).mean()
data['MA25'] = data['close'].rolling(window=5).mean()
# محاسبه MACD و سیگنال
data['MACD'], data['MACD_Signal'] = indicator.calculate_macd(data)

# محاسبه سیگنال‌ها و اضافه کردن به DataFrame
data['Sig_WMA'] = Signal.signal_wma(data, window=14)
data['Sig_RSI'] = Signal.signal_rsi(data, window=14)
data['Sig_MACD'] = Signal.signal_macd(data, fast_window=12, slow_window=26, signal_window=9)
data['Sig_Stochastic'] = Signal.signal_stochastic(data, k_window=14, d_window=3)
data['Sig_ADX'] = Signal.signal_adx(data, window=14)
data['Signal_Buy'] = Signal.signal_buy(data)


# 8. ذخیره داده‌ها در یک فایل CSV جدید
output_file_path = 'output/updated_stock_data3.csv'
data.to_csv(output_file_path, index=False)

print("داده‌ها با موفقیت به روز رسانی شدند و در...", output_file_path)
