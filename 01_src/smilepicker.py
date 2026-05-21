import os
# --- メモリ節約設定 (最優先) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_CPU_AUTOTUNE_STATS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import io
import base64
import numpy as np
import cv2
import gdown
import traceback
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw

# TensorFlowのメモリ消費を最小限に抑える
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

app = Flask(__name__)

# 顔検出器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# =========================================================
# モデル設定（フォルダ階層の最適化）
# =========================================================
# 💡 変更点: 保存および読み込み先を '03_models/' フォルダに変更
MODEL_PATH = '03_models/best_smile_model_v4.keras'
FILE_ID = '12OnNsDlw2cO20HJy941auA74zOAiUuvM'
model = None

def load_saved_model():
    global model
    
    # 💡 変更点: '03_models' フォルダが存在しない場合は自動で作成する安全策を追加
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"📥 モデルが見つからないため、Googleドライブから '{MODEL_PATH}' へダウンロードを開始します...")
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        try:
            gdown.download(url, MODEL_PATH, quiet=False)
        except Exception as e:
            print(f"❌ ダウンロード失敗: {e}")
            return

    if os.path.exists(MODEL_PATH):
        try:
            # 💡 メモリ節約のため、グラフを初期化してからロード
            tf.keras.backend.clear_session()
            # compile=False にすることで、学習用パラメータを読み込まず推論専用にしてメモリを節約
            model = load_model(MODEL_PATH, compile=False)
            print(f"✅ AIモデル '{MODEL_PATH}' のロードに成功しました。")
        except Exception as e:
            print(f"❌ ロード失敗:\n{traceback.format_exc()}")
            model = None

# 起動時にロード
with app.app_context():
    load_saved_model()

# =========================================================
# ルーティングと判定ロジック
# =========================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        load_saved_model()
        if model is None:
            return jsonify({"error": "AIモデルが読み込まれていません。"}), 500
    
    file = request.files.get('file')
    if not file: return jsonify({"error": "ファイルがありません"}), 400
    
    try:
        # 💡 画像サイズをさらに小さく制限（メモリ保護）
        img_pil = Image.open(file.stream).convert('RGB')
        img_pil.thumbnail((600, 600), Image.Resampling.LANCZOS)
        
        img_np = np.array(img_pil)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        draw = ImageDraw.Draw(img_pil)
        any_smile = False

        for (x, y, fw, fh) in faces:
            face_crop = img_pil.crop((x, y, x + fw, y + fh))
            
            # 💡 このモデル(v4)の要求サイズに合わせて224x224にリサイズ
            face_resize = face_crop.resize((224, 224))
            
            x_input = image.img_to_array(face_resize)
            x_input = np.expand_dims(x_input, axis=0) / 255.0

            # 推論実行
            preds = model.predict(x_input, verbose=0)
            smile_score = float(preds[0][1]) 

            is_smile = smile_score > 0.5
            color = (0, 255, 0) if is_smile else (255, 0, 0)
            if is_smile: any_smile = True

            draw.rectangle([x, y, x + fw, y + fh], outline=color, width=3)

        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            "face_count": len(faces),
            "overall_result": "笑顔を検出しました！" if any_smile else "笑顔は見つかりませんでした",
            "image_data": f"data:image/jpeg;base64,{img_str}"
        })
    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "判定に失敗しました"}), 500

if __name__ == '__main__':
    # ポート10000番で起動
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)