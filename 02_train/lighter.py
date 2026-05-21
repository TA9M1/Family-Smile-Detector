import os
import tensorflow as tf

# =========================================================
# パスの設定（整理された 03_models フォルダを基準にする）
# =========================================================
# 📥 読み込み元モデルのパス
input_model_path = '03_models/best_smile_model_v4.keras'

# 📤 変換後のTFLiteモデルの保存先パス
tflite_model_path = '03_models/best_smile_model_v4.tflite'


# 1. 現在のモデルをロード
if os.path.exists(input_model_path):
    print(f"📥 モデル '{input_model_path}' を読み込んでいます...")
    model = tf.keras.models.load_model(input_model_path)
else:
    raise FileNotFoundError(f"❌ 元となるモデルが見つかりません: {input_model_path}\n先に訓練（train）を実行してモデルを生成してください。")

# 2. TFLite Converterの準備
print("⚙️ TFLite形式への変換処理を開始します...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# (オプション) さらなる軽量化：重みを8bit整数に量子化する場合（サイズは約1/4になります）
# 💡 もし本番環境のメモリを限界まで節約したい場合は、下の行のコメントアウト(#)を解除してください
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 3. 変換実行
tflite_model = converter.convert()

# 4. 03_models フォルダの中にファイルとして保存
# 💡 保存先フォルダが万が一ない場合のために自動作成処理を追加
os.makedirs(os.path.dirname(tflite_model_path), exist_ok=True)

with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"✅ 変換完了: '{tflite_model_path}' に軽量化モデルが保存されました。")