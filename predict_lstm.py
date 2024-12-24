import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping


# بارگذاری داده‌ها
data = pd.read_csv('output/updated_stock_data3.csv', parse_dates=['time'], index_col='time')

# حذف رکوردهای خراب
data_cleaned = data.iloc[30:].reset_index(drop=True)

# ادامه پردازش داده‌ها
features = data_cleaned.drop(columns=['state', 'percent', 'new_rank'])
target = data_cleaned[['state', 'percent', 'new_rank']]

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
time_steps = 100
X, y = create_dataset(features_scaled, target, time_steps)

# تبدیل نوع داده‌ها به float32
X = X.astype(np.float32)
y = y.astype(np.float32)

# تقسیم داده‌ها به مجموعه‌های آموزشی و آزمایشی
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ساخت مدل LSTM
# ساخت مدل LSTM
model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(LSTM(32, return_sequences=True))  # return_sequences=True برای لایه‌های میانی
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(LSTM(32, return_sequences=True))  # return_sequences=True برای لایه‌های میانی
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(LSTM(32))  # آخرین لایه LSTM باید return_sequences=False باشد
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dense(3))  # سه خروجی برای state، percent و new_rank

# کامپایل مدل
model.compile(optimizer='adam', loss='mean_squared_error')

# تعریف early_stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# سپس در مدل خود از early_stopping استفاده کنید
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping])

# ارزیابی مدل
loss = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')


# ذخیره مدل
model.save('my_lstm_model.keras')

# پیش‌بینی
predicted = model.predict(X_test)

# تبدیل پیش‌بینی‌ها به DataFrame
predicted_df = pd.DataFrame(predicted, columns=['state', 'percent', 'new_rank'])

# نمایش نتایج
print(predicted_df.head())
