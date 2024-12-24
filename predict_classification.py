import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam


# بارگذاری داده‌ها
data = pd.read_csv('output/updated_stock_data3.csv', parse_dates=['time'], index_col='time')

# حذف رکوردهای خراب
data_cleaned = data.iloc[30:].reset_index(drop=True)

# ایجاد ستون tomaro_state با شیفت دادن ستون state یک روز به عقب
data_cleaned['tomaro_state'] = data_cleaned['state'].shift(-1)

# حذف آخرین ردیف که مقدار tomaro_state ندارد
data_cleaned = data_cleaned[:-1]

# ادامه پردازش داده‌ها
features =data_cleaned[['close']]
# ,'Sig_WMA','Sig_RSI','Sig_MACD','Sig_Stochastic','Sig_ADX','Signal_Buy'
                        # ,'high',	'low',	'open','percent','new_rank'	]]  
                        # 'MA5','MA14','MA25','MACD_Signal',
# فرض کنید data_cleaned دیتافریم شماست
# ابتدا ستون‌های اصلی را انتخاب می‌کنیم

# اضافه کردن ستون‌های جدید با استفاده از np.where
# features['diff_low_high'] = np.where(features['high'] - features['low'] > 0, 1, 0)  # اختلاف low و high
# features['diff_close_open'] = np.where(features['close'] - features['open'] > 1, 1, 0)  # اختلاف close و open
# features['diff_high_close'] = np.where(features['high'] - features['close'] == 0, 1, 0)  # اختلاف high و close
# features['diff_low_open'] = np.where(features['low'] - features['open'] > 1, 1, 0)  # اختلاف low و open
# # اگر نیاز دارید که این ویژگی‌ها را به دیتافریم اصلی اضافه کنید
# data_cleaned = pd.concat([data_cleaned, features[['diff_low_high', 'diff_close_open', 'diff_high_close', 'diff_low_open']]], axis=1)

# نمایش دیتافریم جدید
print(data_cleaned.head())

#data_cleaned.drop(columns=['state', 'percent', 'new_rank', 'tomaro_state'])
target = data_cleaned[['tomaro_state']]  # استفاده از tomaro_state به عنوان لیبل

# تبدیل مقادیر غیر عددی به عددی
label_encoder = LabelEncoder()
target['tomaro_state'] = label_encoder.fit_transform(target['tomaro_state'])

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

# ساخت مدل
model = Sequential()

# لایه اول LSTM
model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# لایه دوم LSTM
model.add(LSTM(50, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# لایه سوم LSTM
model.add(LSTM(25, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# لایه آخر LSTM
model.add(LSTM(10))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# لایه‌های Dense
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # برای طبقه‌بندی باینری

# کامپایل مدل با نرخ یادگیری تنظیم شده
model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
# تعریف early_stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# آموزش مدل
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64)

# چاپ خلاصه مدل
model.summary()

# ارزیابی مدل
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')




# ذخیره مدل
model.save('my_gru_model.keras')

# پیش‌بینی
predicted = model.predict(X_test)

# تبدیل پیش‌بینی‌ها به DataFrame
predicted_df = pd.DataFrame(predicted, columns=['tomaro_state_prediction'])
predicted_df['tomaro_state_prediction'] = (predicted_df['tomaro_state_prediction'] > 0.5).astype(int)  # تبدیل به 0 و 1

# نمایش نتایج
print(predicted_df.head(50))
