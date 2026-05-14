import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw

# --- 1. MediaPipe のインポート (標準的な形式) ---
import mediapipe as mp

try:
    # 以前の 'python.solutions' ではなく、標準の 'solutions' を使用
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    print("✅ MediaPipeの顔検出モジュールを正常に読み込みました。")
except Exception as e:
    print(f"❌ MediaPipeの読み込みに失敗しました: {e}")
    face_detection = None

# iPhoneのHEIC形式への対応
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("⚠️ pillow-heifがインストールされていないため、HEICは利用できません。")

app = Flask(__name__)

# --- 2. 自作AIモデルの読み込み ---
MODEL_PATH = 'best_smile_model.keras'
model = None

def load_saved_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            # 推論専用のため compile=False
            model = load_model(MODEL_PATH, compile=False)
            print(f"✅ AIモデル '{MODEL_PATH}' のロードに成功しました。")
        except Exception as e:
            print(f"❌ モデル読み込みエラー: {e}")
    else:
        print(f"⚠️ {MODEL_PATH} が見つかりません。パスを確認してください。")

load_saved_model()

# --- 3. ルーティングと判定ロジック ---

@app.route('/')
def index():
    """トップ画面(templates/index.html)を表示"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """画像を受け取り、笑顔判定と加工を行って返す"""
    if model is None:
        return jsonify({"error": "AIモデルが読み込まれていません。"}), 500
    if face_detection is None:
        return jsonify({"error": "顔検出エンジン(MediaPipe)が起動していません。"}), 500
    
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "ファイルがアップロードされていません。"}), 400
    
    try:
        # 画像の読み込み
        img_pil = Image.open(file.stream).convert('RGB')
        img_np = np.array(img_pil)
        h, w, _ = img_np.shape
        draw = ImageDraw.Draw(img_pil)

        # MediaPipeによる顔検出実行
        results = face_detection.process(img_np)
        
        face_count = 0
        any_smile = False

        if results.detections:
            face_count = len(results.detections)
            for detection in results.detections:
                # 座標の取得
                bbox = detection.location_data.relative_bounding_box
                xmin, ymin = int(bbox.xmin * w), int(bbox.ymin * h)
                width, height = int(bbox.width * w), int(bbox.height * h)

                # 顔部分を切り抜いて前処理
                face_crop = img_pil.crop((max(0, xmin), max(0, ymin), min(w, xmin+width), min(h, ymin+height)))
                face_resize = face_crop.resize((128, 128))
                x = image.img_to_array(face_resize)
                x = np.expand_dims(x, axis=0)
                x /= 255.0

                # 自作モデルによる推論
                preds = model.predict(x)
                smile_score = float(preds[0][1])

                # 判定結果に基づく描画設定
                is_smile = smile_score > 0.5
                color = (0, 255, 0) if is_smile else (255, 0, 0) # 緑(笑顔) or 赤(真顔)
                if is_smile: any_smile = True

                label = f"{'Smile' if is_smile else 'Neutral'}: {smile_score*100:.1f}%"
                
                # 枠と文字の描き込み
                line_w = max(3, int(w / 150))
                draw.rectangle([xmin, ymin, xmin + width, ymin + height], outline=color, width=line_w)
                draw.text((xmin, max(0, ymin - 25)), label, fill=color)

        # 加工した画像をBase64形式の文字列に変換
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            "face_count": face_count,
            "overall_result": "笑顔を検出しました！😊" if any_smile else "笑顔は見つかりませんでした😐",
            "image_data": f"data:image/jpeg;base64,{img_str}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # ポート10000番で起動
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
    




import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# 1. 評価の準備（順番を固定）
validation_generator.shuffle = False
validation_generator.reset()

# 2. まとめて予測
predictions = model.predict(validation_generator)
y_pred = np.argmax(predictions, axis=1) # 予測したクラス
y_true = validation_generator.classes   # 本当のクラス

# 3. 混同行列の計算
cm = confusion_matrix(y_true, y_pred)
print("\n=== 混同行列 ===")
print(cm)
print(f"クラス名: {validation_generator.class_indices}")

# 4. 「自信満々に間違えた」画像を探す
# 自信度（確率）
confidences = np.max(predictions, axis=1)
# 間違えたデータのインデックス
errors = np.where(y_pred != y_true)[0]

print(f"\n間違い数: {len(errors)} / {len(y_true)}")

# 間違えた画像のうち、最初の3枚だけ「AIが何%の自信で間違えたか」表示
for i in errors[:3]:
    print(f"画像Index {i}: 正解={y_true[i]}, 予測={y_pred[i]}, 自信度={confidences[i]:.2f}")