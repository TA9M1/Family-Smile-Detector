import os
import numpy as np
import matplotlib.pyplot as plt

# Keras 3 / TensorFlow のインポート
import keras
from keras import layers, models, optimizers, callbacks
from keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================================================
# 1. ネットワークの構築
# =========================================================
# 入力サイズ 128x128, 3チャンネル(RGB)
input_tensor = layers.Input(shape=(128, 128, 3))

# 学習済みのVGG16をロード（全結合層なし）
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# カスタム層の追加
x = vgg16.output
x = layers.GlobalAveragePooling2D()(x) # 特徴マップを1次元に圧縮 🧠
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
# 出力層（2クラス：笑顔・それ以外）
predictions = layers.Dense(2, activation='softmax')(x)

model = models.Model(inputs=vgg16.input, outputs=predictions)

# =========================================================
# 2. ファインチューニングの設定
# =========================================================
# 前方の層をフリーズし、後方の層のみ再学習させる
for layer in vgg16.layers[:11]:
    layer.trainable = False
for layer in vgg16.layers[11:]:
    layer.trainable = True

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(learning_rate=1e-5),
    metrics=['accuracy']
)

# =========================================================
# 3. データ拡張と読み込み
# =========================================================
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2 # 20%をテスト用に分割 📁
)

train_generator = datagen.flow_from_directory(
    'data',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'data',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# =========================================================
# 4. 学習の実行
# =========================================================
# 💡 変更点: 保存先を '03_models/' フォルダに変更
checkpoint = callbacks.ModelCheckpoint(
    filepath='03_models/best_smile_model.keras',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

print("\n--- Keras 3環境での学習を開始します ---")

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=100,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

# 💡 変更点: 保存先を '03_models/' フォルダに変更
model.save('03_models/best_smile_model.keras')
print("\n✅ モデルを '03_models/best_smile_model.keras' として保存しました。")