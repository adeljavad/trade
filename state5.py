import pandas as pd
import numpy as np
# adel test commit
# 1. خواندن داده‌های CSV
file_path = 'stock_data.csv'  # مسیر فایل CSV
data = pd.read_csv(file_path)

# 2. اضافه کردن ستون‌های جدید با نوع داده مناسب
data['state'] = pd.Series(dtype='object')  # تعیین نوع داده به object
data['percent'] = np.nan
data['new_rank'] = np.nan  # اضافه کردن ستون new_rank
data['RTP'] = np.nan  # اضافه کردن ستون RTP
data['new_rtp'] = np.nan  # اضافه کردن ستون new_rtp

# تابع برای تعیین وضعیت (خرید یا فروش)
def determine_state(current_price, previous_price):
    if current_price > previous_price:
        return 'buy'
    else:
        return 'sell'

# تابع برای محاسبه درصد تغییر
def calculate_percent(current_price, previous_price):
    if previous_price != 0:  # جلوگیری از تقسیم بر صفر
        return ((current_price - previous_price) / previous_price) * 100
    return np.nan  # اگر قیمت قبلی صفر باشد

# تابع برای محاسبه رنک
def calculate_rank(data, start_index, streak_counter, current_state):
    for j in range(start_index, start_index + streak_counter):
        if current_state == 'buy':
            data.at[j, 'new_rank'] = streak_counter - (j - start_index)
        else:
            data.at[j, 'new_rank'] = -(streak_counter - (j - start_index))


def calculate_rtp(data, start_index, streak_counter, current_state):
    total_percent = 0  # جمع کل درصد
    k=streak_counter+start_index
    for j in range(start_index, start_index + streak_counter):
        k -= 1
        if current_state == 'buy':
            total_percent += data.at[j, 'percent']
        else:
            total_percent -= data.at[j, 'percent']
        
        # ذخیره در ستون new_rtp برای هر ردیف
        data.at[j, 'new_rtp1'] = total_percent  
        data.at[k, 'new_rtp'] = total_percent  
        data.at[j, 'max'] = data.at[1 ,'new_rtp']-data.at[j-1 ,'new_rtp']
    
    total_percent = 0  # جمع کل درصد
    k = streak_counter+start_index
    for j in range(start_index, start_index + streak_counter):
        k -= 1
        if current_state == 'buy':
            total_percent += data.at[k, 'percent']-data.at[j, 'percent']
        else:
            total_percent -= data.at[j, 'percent']
        
        # ذخیره در ستون new_rtp برای هر ردیف
        data.at[j, 'new_rtp3'] = total_percent  
        data.at[k, 'new_rtp2'] = total_percent  
        

        
# تابع برای محاسبه RTP
def calculate_running_total_percent(data):
    total_percent = 0  # جمع کل درصد
    total_list = []  # لیست برای نگهداری مقادیر جمع شده

    # محاسبه جمع کل درصدها به‌صورت معکوس بر اساس نوع buy و sell
    for i in range(len(data) - 1, -1, -1):
        if data.at[i, 'state'] == 'buy':
            total_percent += data.at[i, 'percent']
        else:
            total_percent -= data.at[i, 'percent']  # برای sell هم درصد را جمع می‌کنیم
        total_list.append(total_percent)  # افزودن به لیست

    # معکوس کردن لیست برای قرار دادن در DataFrame
    total_list.reverse()
    data['RTP'] = total_list  # ذخیره در ستون RTP

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

# 4. اجرای تابع محاسبه وضعیت، درصد و new_rank
calculate_state_and_new_rank(data)

# 5. محاسبه RTP
calculate_running_total_percent(data)

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

calculate_new_rtp(data)

# 7. معکوس کردن ترتیب new_rtp
# data['new_rtp'] = data['new_rtp'][::-1].reset_index(drop=True)

# 8. ذخیره داده‌ها در یک فایل CSV جدید
output_file_path = 'updated_stock_data2.csv'
data.to_csv(output_file_path, index=False)

print("داده‌ها با موفقیت به روز رسانی شدند و در فایل جدید ذخیره شدند.")
