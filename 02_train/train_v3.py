import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# =========================================================
# 設定変更: 解像度を VGG16 の標準 224 に変更
# =========================================================
IMG_SIZE = 224
BATCH_SIZE = 16  # 解像度アップに伴い、メモリ節約のためバッチサイズを小さく調整

# =========================================================
# 1. ネットワークの構築
# =========================================================
input_tensor = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

x = vgg16.output
x = layers.GlobalAveragePooling2D()(x)

# Dropoutを強化 (0.6 -> 0.7) し、L2正則化も継続
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = layers.Dropout(0.7)(x) 

# 中間層にも Dropout (0.5) を追加して過学習を徹底ガード
x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = layers.Dropout(0.5)(x)

predictions = layers.Dense(2, activation='softmax')(x)
model = models.Model(inputs=vgg16.input, outputs=predictions)

# =========================================================
# 2. ファインチューニングの設定
# =========================================================
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
# 💡 カレントディレクトリ（プロジェクトのルート）を基準にするため 'data' のまま、
# または環境に合わせて '01_input/data' などに適宜書き換えてください。
DATA_DIR = 'data' 

datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, 
    class_mode='categorical', subset='training'
)

validation_generator = datagen.flow_from_directory(
    DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, 
    class_mode='categorical', subset='validation', shuffle=False 
)

# =========================================================
# 4. 学習の実行
# =========================================================
# 💡 変更点: 保存先を '03_models/' フォルダに変更
checkpoint = callbacks.ModelCheckpoint(
    filepath='03_models/best_smile_model_v4.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'
)
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', patience=25, restore_best_weights=True # 耐性を少し増やしてじっくり学習
)

print(f"\n--- 学習開始 (解像度: {IMG_SIZE}, バッチサイズ: {BATCH_SIZE}) ---")
history = model.fit(
    train_generator, epochs=150, validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping], verbose=1
)

# 💡 変更点: 保存先を '03_models/' フォルダに変更
model.save('03_models/final_smile_model_v4.keras')
print("\n✅ モデルを '03_models/final_smile_model_v4.keras' に保存しました。")

# =========================================================
# 5. 精度・損失グラフの保存
# =========================================================
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    # 💡 変更点: 保存先を '04_output/' フォルダに変更
    plt.savefig('04_output/learning_history_v4.png')
    print("\n✅ 学習曲線を '04_output/learning_history_v4.png' に保存しました。")
    plt.close()

plot_history(history)

# =========================================================
# 6. 詳細評価
# =========================================================
print("\n--- 詳細評価を開始します ---")
validation_generator.reset()
predictions = model.predict(validation_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

print("\n【混同行列】")
print(confusion_matrix(y_true, y_pred))
print("\n【分類レポート】")
print(classification_report(y_true, y_pred, target_names=class_labels))

# 苦手分析画像の保存
confidences = np.max(predictions, axis=1)
errors = np.where(y_pred != y_true)[0]

if len(errors) > 0:
    error_indices_sorted = errors[np.argsort(confidences[errors])][::-1]
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(error_indices_sorted[:5]):
        if i >= len(error_indices_sorted): break
        img_path = validation_generator.filepaths[idx]
        img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.title(f"True:{class_labels[y_true[idx]]}\nPred:{class_labels[y_pred[idx]]}\nConf:{confidences[idx]:.2f}")
        plt.axis('off')
        
    # 💡 変更点: 保存先を '04_output/' フォルダに変更
    plt.savefig('04_output/worst_errors_v4.png')
    print("✅ 苦手分析画像を '04_output/worst_errors_v4.png' に保存しました。")
    plt.close()