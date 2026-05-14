import os
# --- メモリ節約のための設定 (最優先) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'      # ログ出力を最小限にしてメモリ節約
os.environ['TF_FORCE_CPU_AUTOTUNE_STATS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'           # 並列処理を制限してメモリ消費を抑える

import io
import base64
import numpy as np
import cv2
import gdown
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw

# --- TensorFlowの追加メモリ制限 ---
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# HEIC形式への対応
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("⚠️ pillow-heifがインストールされていないため、HEICは利用できません。")

app = Flask(__name__)

# --- 顔検出の設定 (OpenCV Haar Cascade) ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 2. 自作AIモデルの読み込みとダウンロード設定 ---
MODEL_PATH = 'best_smile_model_v4.keras'

# 【重要】ここにあなたのGoogleドライブのファイルIDを入れてください
FILE_ID = '12OnNsDlw2cO20HJy941auA74zOAiUuvM'

model = None

def load_saved_model():
    global model
    
    # 1. モデルファイルが存在しない場合はダウンロードを実行
    if not os.path.exists(MODEL_PATH):
        print("📥 モデルファイルが見つかりません。Googleドライブからダウンロードを開始します...")
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        try:
            gdown.download(url, MODEL_PATH, quiet=False)
        except Exception as e:
            print(f"❌ ダウンロードに失敗しました: {e}")
            return

    # 2. モデルの読み込み
    if os.path.exists(MODEL_PATH):
        try:
            # メモリ節約のためセッションをクリアしてからロード
            tf.keras.backend.clear_session()
            model = load_model(MODEL_PATH, compile=False)
            print(f"✅ AIモデル '{MODEL_PATH}' のロードに成功しました。")
        except Exception as e:
            print(f"❌ モデル読み込みエラー: {e}")
    else:
        print(f"⚠️ {MODEL_PATH} の準備ができていないため、判定は利用できません。")

# --- 3. ルーティングと判定ロジック ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "AIモデルが読み込まれていません。サーバーの起動ログを確認してください。"}), 500
    
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "ファイルがアップロードされていません。"}), 400
    
    try:
        # 画像の読み込み
        img_pil = Image.open(file.stream).convert('RGB')
        
        # 巨大画像対策（さらに小さめの1000pxに制限してメモリを保護）
        max_limit = 1000
        if max(img_pil.size) > max_limit:
            img_pil.thumbnail((max_limit, max_limit), Image.Resampling.LANCZOS)

        img_np = np.array(img_pil)
        
        # 顔検出用グレースケール変換
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # 顔検出
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        draw = ImageDraw.Draw(img_pil)
        face_count = len(faces)
        any_smile = False

        for (x, y, fw, fh) in faces:
            # 顔切り抜き
            face_crop = img_pil.crop((max(0, x), max(0, y), min(w, x + fw), min(h, y + fh)))
            
            # v4モデル用に224x224へリサイズ
            face_resize = face_crop.resize((224, 224))
            
            # 推論用前処理
            x_input = image.img_to_array(face_resize)
            x_input = np.expand_dims(x_input, axis=0)
            x_input /= 255.0

            # 笑顔判定
            preds = model.predict(x_input)
            smile_score = float(preds[0][1]) 

            is_smile = smile_score > 0.5
            color = (0, 255, 0) if is_smile else (255, 0, 0)
            if is_smile: any_smile = True

            # 枠とラベルの描画
            label = f"{'Smile' if is_smile else 'Neutral'}: {smile_score*100:.1f}%"
            line_w = max(2, int(max(img_pil.size) / 250))
            draw.rectangle([x, y, x + fw, y + fh], outline=color, width=line_w)
            draw.text((x, max(0, y - 20)), label, fill=color)

        # 結果画像のBase64化
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG", quality=80) # 画質を少し下げて転送速度を稼ぐ
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            "face_count": face_count,
            "overall_result": f"{face_count}人の顔を検出！ " + ("笑顔をキャッチしました！😊" if any_smile else "笑顔は見つかりませんでした😐"),
            "image_data": f"data:image/jpeg;base64,{img_str}"
        })

    except Exception as e:
        return jsonify({"error": f"サーバーエラー: {str(e)}"}), 500

if __name__ == '__main__':
    # 起動時にモデルを準備
    load_saved_model()
    
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)