import os
# --- メモリ節約設定 (最優先) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import io
import base64
import numpy as np
import cv2
import traceback
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw

# =========================================================
# Flask アプリ初期化設定 (修正箇所)
# =========================================================
# main.pyから見て、1つ上の階層（..）にある「templates」と「static」を指定します。
app = Flask(
    __name__, 
    template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), '..', 'static')
)

# OpenCVの顔検出器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# =========================================================
# TFLite モデル設定
# =========================================================
# 💡 変更点: 保存先を整理した '03_models/' の tflite パスに指定
MODEL_PATH = '03_models/best_smile_model_v4.tflite'
interpreter = None
input_details = None
output_details = None

def load_tflite_model():
    global interpreter, input_details, output_details
    if os.path.exists(MODEL_PATH):
        try:
            # 💡 TFLite専用の読み込み方法（インプリターの起動）
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            
            # 入出力テンソルの情報を取得（データの型や形を確認するため）
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(f"✅ TFLiteモデル '{MODEL_PATH}' のロードに成功しました。")
        except Exception as e:
            print(f"❌ TFLiteモデル読み込みエラー: {e}")
    else:
        print(f"⚠️ {MODEL_PATH} が見つかりません。先に 02_train/lighter.py を実行して生成してください。")

# 起動時にロード
with app.app_context():
    load_tflite_model()

# =========================================================
# ルーティングと判定ロジック
# =========================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global interpreter, input_details, output_details
    if interpreter is None:
        return jsonify({"error": "TFLiteモデルが読み込まれていません。"}), 500
    
    file = request.files.get('file')
    if not file: return jsonify({"error": "ファイルがありません"}), 400
    
    try:
        # 画像サイズを制限（メモリ保護）
        img_pil = Image.open(file.stream).convert('RGB')
        img_pil.thumbnail((600, 600), Image.Resampling.LANCZOS)
        
        img_np = np.array(img_pil)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        draw = ImageDraw.Draw(img_pil)
        any_smile = False

        for (x, y, fw, fh) in faces:
            face_crop = img_pil.crop((x, y, x + fw, y + fh))
            # VGG16用の解像度 224x224 にリサイズ
            face_resize = face_crop.resize((224, 224))
            
            x_input = image.img_to_array(face_resize)
            x_input = np.expand_dims(x_input, axis=0) / 255.0
            x_input = x_input.astype(np.float32) # TFLiteは型に厳密なため float32 に明示的に変換

            # 💡 TFLiteでの推論実行手順
            interpreter.set_tensor(input_details[0]['index'], x_input)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])
            
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
            "overall_result": "笑顔を検出しました！😊" if any_smile else "笑顔は見つかりませんでした😐",
            "image_data": f"data:image/jpeg;base64,{img_str}"
        })
    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "判定に失敗しました"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)