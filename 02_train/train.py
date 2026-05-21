import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# 💡 注意: この評価コードを動かす前に、前段で 'model' がロードされている必要があります。
#（例: model = keras.models.load_model('03_models/best_smile_model.keras') など）

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