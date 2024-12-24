from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# بارگذاری داده‌ها
data = pd.read_csv('output/updated_stock_data2.csv', parse_dates=['time'], index_col='time')


# انتخاب ویژگی‌ها و هدف
features = data.drop(columns=['state', 'percent', 'new_rank'])
target = data[['state', 'percent', 'new_rank']]

# تبدیل مقادیر غیر عددی به عددی
label_encoder = LabelEncoder()
target['state'] = label_encoder.fit_transform(target['state'])

# نرمال‌سازی داده‌ها
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# تبدیل داده‌ها به توالی‌های زمانی
def create_dataset(features, target, time_steps=1):
    X, y = [], []
    for i in range(len(features) - time_steps):
        X.append(features[i:(i + time_steps)])
        y.append(target.iloc[i + time_steps].values)
    return np.array(X), np.array(y)

# تعریف پارامترها
time_steps = 10
X, y = create_dataset(features_scaled, target, time_steps)

# تبدیل نوع داده‌ها به float32
X = X.astype(np.float32)
y = y.astype(np.float32)

# تقسیم داده‌ها به مجموعه‌های آموزشی و آزمایشی
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# بارگذاری مدل
loaded_model = load_model('my_lstm_model.keras')

# استفاده از مدل بارگذاری شده برای پیش‌بینی
predicted_loaded = loaded_model.predict(X_test)

# تبدیل پیش‌بینی‌ها به DataFrame
predicted_loaded_df = pd.DataFrame(predicted_loaded, columns=['state', 'percent', 'new_rank'])

# نمایش نتایج
print(predicted_loaded_df.head())
