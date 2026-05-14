import os
import io
import base64
import numpy as np
import cv2  # OpenCVを使用
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw

# --- 1. 顔検出の設定 (OpenCV Haar Cascade) ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# HEIC形式への対応
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("⚠️ pillow-heifがインストールされていないため、HEICは利用できません。")

app = Flask(__name__)

# --- 2. 自作AIモデルの読み込み設定 ---
# v4モデルを指定。Renderにデプロイする際は、このファイルがルートディレクトリにある必要があります。
MODEL_PATH = 'best_smile_model_v4.keras'
model = None

def load_saved_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            # compile=Falseにすることで、学習時の環境に依存せずモデル構造と重みのみを読み込みます
            model = load_model(MODEL_PATH, compile=False)
            print(f"✅ AIモデル '{MODEL_PATH}' のロードに成功しました。")
        except Exception as e:
            print(f"❌ モデル読み込みエラー: {e}")
    else:
        print(f"⚠️ {MODEL_PATH} が見つかりません。")

# --- 3. ルーティングと判定ロジック ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "AIモデルが読み込まれていません。"}), 500
    
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "ファイルがアップロードされていません。"}), 400
    
    try:
        # 画像の読み込み
        img_pil = Image.open(file.stream).convert('RGB')
        
        # --- 【追加】巨大な画像への対策 ---
        # 処理速度とメモリ節約のため、長辺を最大1200pxにリサイズ
        max_limit = 1200
        if max(img_pil.size) > max_limit:
            img_pil.thumbnail((max_limit, max_limit), Image.Resampling.LANCZOS)
            print(f"📏 全体画像を {img_pil.size} にリサイズしました。")

        img_np = np.array(img_pil)
        
        # OpenCV用にグレースケール変換（顔検出用）
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # 顔検出の実行
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        draw = ImageDraw.Draw(img_pil)
        face_count = len(faces)
        any_smile = False

        for (x, y, fw, fh) in faces:
            # 顔部分を切り抜き
            face_crop = img_pil.crop((max(0, x), max(0, y), min(w, x + fw), min(h, y + fh)))
            
            # --- 【修正】モデルv4の期待値 224x224 にリサイズ ---
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

            # 描画処理：画像サイズに応じて線の太さを動的に変更
            label = f"{'Smile' if is_smile else 'Neutral'}: {smile_score*100:.1f}%"
            line_w = max(2, int(max(img_pil.size) / 250))
            
            draw.rectangle([x, y, x + fw, y + fh], outline=color, width=line_w)
            # 文字位置を枠の少し上に配置
            draw.text((x, max(0, y - 20)), label, fill=color)

        # 結果画像をJPEG形式でBase64エンコード
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            "face_count": face_count,
            "overall_result": f"{face_count}人の顔を検出！ " + ("笑顔をキャッチしました！😊" if any_smile else "笑顔は見つかりませんでした😐"),
            "image_data": f"data:image/jpeg;base64,{img_str}"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"サーバーエラー: {str(e)}"}), 500

if __name__ == '__main__':
    # 起動時に一度だけモデルをロード
    load_saved_model()
    
    # Render等の環境変数を優先し、デフォルト10000番ポートで起動
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 SmilePicker Server is running on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)