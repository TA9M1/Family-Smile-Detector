import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

# =========================================================
# 1. ネットワークの構築 (解像度を128にアップ)
# =========================================================
# 64x64から128x128に上げることで、目元や口元の特徴を掴みやすくします
input_tensor = Input(shape=(128, 128, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5)) # 強力に暗記を阻止
top_model.add(Dense(128, activation='relu'))
top_model.add(Dropout(0.3))
top_model.add(Dense(2, activation='softmax'))

model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

# =========================================================
# 2. ファインチューニングの設定
# =========================================================
for layer in model.layers[:11]:
    layer.trainable = False
for layer in model.layers[11:]:
    layer.trainable = True

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate=1e-5),
              metrics=['accuracy'])

# =========================================================
# 3. コールバックの設定 (早期終了と学習率調整を追加)
# =========================================================
# 最高精度のモデルを保存
checkpoint = ModelCheckpoint(
    filepath='best_smile_model.keras',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

# 精度が改善しなくなったら自動で止める (副作用防止)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20, # 20エポック改善しなければ終了
    restore_best_weights=True # 最も良かった時の重みに戻す
)

# 精度が停滞したら学習率を下げる
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=7,
    min_lr=1e-7
)

# =========================================================
# 4. データ拡張と読み込み (解像度を合わせる)
# =========================================================
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,       # 少しだけ回転を強める
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    'data',
    target_size=(128, 128),  # 128に合わせる
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'data',
    target_size=(128, 128),  # 128に合わせる
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# =========================================================
# 5. 学習の実行
# =========================================================
print("\n--- 改善版：過学習対策済みモデルでの学習を開始します ---")

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=200,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint, early_stopping, reduce_lr], # 3つの薬を処方
    verbose=1
)

# =========================================================
# 6. 結果の可視化
# =========================================================
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Test Acc')
plt.axhline(y=0.8, color='r', linestyle='--', label='Goal 80%')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.show()

print(f"\n★今回の最高精度: {max(history.history['val_accuracy'])*100:.2f}%")