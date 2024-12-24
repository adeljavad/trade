import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# بارگذاری داده‌ها
data = pd.read_csv('output/updated_stock_data3.csv')

# فرض کنید که شما یک ستون 'state' دارید که وضعیت فردا را نشان می‌دهد
# data = data[['time','close','tomaro_state','state']]
data['tomaro_state'] = data['state'].shift(-1)
data = data[:-1]  # حذف آخرین ردیف که مقدار ندارد

# تبدیل مقادیر غیر عددی به عددی
label_encoder = LabelEncoder()
data['tomaro_state'] = label_encoder.fit_transform(data['tomaro_state'])

# پر کردن مقادیر خالی یا حذف آن‌ها
data.fillna(0, inplace=True)  # پر کردن با صفر

# بررسی نوع داده‌ها
print(data.dtypes)

# حذف ستون‌های غیر عددی (در صورت نیاز)
data = data.select_dtypes(include=[np.number])  # فقط ستون‌های عددی را نگه دارید

# بررسی تعداد نمونه‌ها
print(f"Total samples before slicing: {len(data)}")

# آماده‌سازی داده‌ها
time_steps = 30  # کاهش طول توالی
X, y = [], []
for i in range(len(data) - time_steps):
    X.append(data.iloc[i:i + time_steps].values)
    y.append(data['tomaro_state'].iloc[i + time_steps])

X = np.array(X, dtype=np.float32)  # تغییر به np.float32 برای دقت بیشتر
y = np.array(y, dtype=np.float32)

# بررسی تعداد نمونه‌ها بعد از آماده‌سازی
print(f"Number of samples: {len(X)}")

# تقسیم داده‌ها به مجموعه‌های آموزشی و آزمایشی
if len(X) > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # تعریف تعداد کلاس‌ها
    num_classes = 2

    # تعریف مدل
    def make_model(input_shape):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.ReLU()(conv1)

        conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.ReLU()(conv2)

        conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.ReLU()(conv3)

        gap = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

        return keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    model = make_model(input_shape=X_train.shape[1:])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    keras.utils.plot_model(model, show_shapes=True)
    epochs = 500
    batch_size = 1024

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "best_model.keras", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )

    # پیش‌بینی
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)

    # نمایش نتایج
    print(predicted_classes)
else:
    print("No samples available for training.")
