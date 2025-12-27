# CSIRO - Image2Biomass Prediction

## Directory Structure
```text
.
├── src/                        # プロジェクトのコアロジック
│   ├── config.py               # パスやハイパーパラメータの集中管理
│   ├── data/                   # データの読み込み・前処理
│   │   └── preprocessing.py    # 画像分割やピボット処理
│   ├── features/               # 特徴量エンジニアリング
│   │   ├── embedding.py        # AIモデルによる画像ベクトル化
│   │   └── supervised.py       # 次元圧縮 (PCA/PLS)
│   ├── models/                 # 学習モデルの定義
│   │   └── tabular.py          # 表データ用学習ラッパー
│   └── utils/                  # 共通ユーティリティ
│       └── metrics.py          # スコア計算と後処理ロジック
├── experiments/                # 実行用スクリプト
│   └── train_tabular.py        # 学習・検証のメインエントリー
├── outputs/                    # 学習済みモデルや予測結果の保存先
└── README.md                   # このドキュメント
```

## 各ディレクトリ・ファイルの解説
src/ (Source Code)
システムの核となるロジックをモジュール化したディレクトリです。

config.py: 全ての実験で共通して使用する「定数」を管理します。ここを書き換えるだけで、全工程のデバイス（CPU/GPU）やファイルパスを一括変更できます。

data/preprocessing.py: このコンペ特有の「1画像に対して5つのターゲットがある」形式を扱いやすく整理します。また、高解像度画像をモデルが読み込めるパッチ状に切断するロジックも含みます。

features/: 画像を「知識」に変える工程です。

embedding.py: 事前学習済みAIを使い、画像を数値ベクトルに変換します。

supervised.py: 膨大な次元のベクトルから、予測に有効な成分だけを濃縮（PLS/PCA）します。

models/tabular.py: 5つの成分を個別に、あるいは同時に学習させるための汎用的な仕組みを提供します。CatBoostやLightGBMなどを差し替えやすく設計されています。

utils/metrics.py: このコンペで最も重要な**「物理的な一貫性（各成分の和がTotalと一致すること）」**を保証するための数式補正（Reconciliation）を実装しています。

experiments/ (Execution)
実際に手を動かしてモデルを作るための「実行ボタン」の役割を果たすスクリプト群です。

train_tabular.py: データの読み込みから最終スコアの表示までを一本道で実行します。コマンドラインから python experiments/train_tabular.py で実行可能です。

## 基本的な処理の流れ
1. データ整形: preprocessing.py で画像をタイル状に分割。

2. 特徴量抽出: embedding.py でAIによる画像ベクトルを生成。

3. 特徴量圧縮: supervised.py で目的変数に相関の高い成分を抽出。

4. 学習・検証: tabular.py で交差検証（CV）を実施。

5. 一貫性補正: metrics.py で予測値の矛盾を修正し、最終スコアを算出
---

