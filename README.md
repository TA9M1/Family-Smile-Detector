# Family Smile Detector (SmilePicker) 😊

AIが写真の中から笑顔を自動検出し、その「笑顔度」をスコア化するウェブアプリケーションです。
家族や友人との集合写真から、最高の1枚を選ぶお手伝いをします。

## 🌟 主な機能
- **複数人の同時判定**: OpenCVを使用して写真内の顔をすべて検出し、一人ひとりの笑顔を判定します。
- **高精度AIモデル**: TensorFlow/Kerasで自作したCNNモデル（v4）を使用し、解像度224x224で詳細に解析します。
- **巨大画像対応**: 高解像度の写真も自動でリサイズして処理するため、スマホで撮った写真もそのままアップロード可能です。
- **直感的なUI**: 判定結果を画像上に枠とスコアで表示します。

## 🚀 デプロイ先 (Live Demo)
[ReFamily-Smile-Detector](https://family-smile-detector.onrender.comのRenderのURL)

## 🛠️ 使用技術
- **Backend**: Python 3.11, Flask
- **AI/ML**: TensorFlow 2.x, Keras, OpenCV (Haar Cascades)
- **Frontend**: HTML5, CSS3 (Modern Responsive Design)
- **Deployment**: Render

## 📦 インストールと実行方法（ローカル環境）

1. リポジトリをクローン
```bash
git clone [https://github.com/TA9M1/Family-Smile-Detector.git](https://github.com/TA9M1/Family-Smile-Detector.git)
cd Family-Smile-Detector
必要なライブラリをインストール

Bash
pip install -r requirements.txt
学習済みモデルの配置
Googleドライブから best_smile_model_v4.keras をダウンロードし、ルートディレクトリに配置してください。

アプリの起動

Bash
python smilepicker.py
http://localhost:10000 にアクセスして利用できます。

📊 モデルの学習について
本アプリで使用しているモデルは、約900枚の顔画像データを使用して学習されました。

入力サイズ: 224x224 (RGB)

検証正解率 (Validation Accuracy): 約83%

詳細: 学習の記録や考察についてはこちらのブログ記事をご覧ください。

📄 ライセンス
MIT License