"""ついに全てのパーツが揃いました！最後の仕上げとして、これらを統合して動かす experiments/train_tabular.py を作成します。

このスクリプトは、データ読み込みから特徴量抽出、学習、スコア計算、そして結果の保存までを一本道で実行します。これが完成すれば、VS Codeのターミナルからコマンド一つで実験を回せる「プロのMLエンジニア」の環境が完成します。"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# プロジェクトルートをパスに追加（srcをインポート可能にする）
sys.path.append(os.curdir)

from src.config import CFG
from src.data.preprocessing import pivot_table, preprocess_image_path
from src.features.embedding import compute_embeddings, generate_semantic_features
from src.features.supervised import SupervisedEmbeddingEngine
from src.models.tabular import run_tabular_cv
from src.utils.metrics import competition_metric, post_process_biomass
from src.utils.logger import seeding, flush

def main():
    # 1. 環境準備
    seeding(CFG.SEED)
    os.makedirs("outputs", exist_ok=True)
    
    # 2. データの読み込みと整形
    print("Loading data...")
    train_raw = pd.read_csv(CFG.TRAIN_CSV)
    train_df = pivot_table(train_raw)
    train_df = preprocess_image_path(train_df, CFG.IMAGE_DIR)
    
    # 簡単なFold分割（もしCSVにfold列がない場合）
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    train_df['fold'] = -1
    for i, (_, val_idx) in enumerate(kf.split(train_df)):
        train_df.loc[val_idx, 'fold'] = i

    # 3. 特徴量抽出 (画像 -> ベクトル)
    # ※ 本来は一度計算したらキャッシュ(Save/Load)するのが望ましいです
    siglip_path = "/kaggle/input/google-siglip-so400m-patch14-384" # 環境に合わせて修正
    if not os.path.exists("outputs/train_embeddings.npy"):
        emb_df = compute_embeddings(train_df, model_path=siglip_path)
        X_emb = emb_df.filter(like="emb").values
        np.save("outputs/train_embeddings.npy", X_emb)
    else:
        X_emb = np.load("outputs/train_embeddings.npy")
        
    # セマンティック特徴量の生成
    X_semantic = generate_semantic_features(X_emb, model_path=siglip_path)
    
    # 4. モデル学習 (Cross Validation)
    print("Starting Tabular Training (CatBoost)...")
    feat_engine = SupervisedEmbeddingEngine(n_pca=0.80, n_pls=8, n_gmm=6)
    model_instance = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        verbose=0,
        random_seed=CFG.SEED
    )
    
    oof_preds = run_tabular_cv(
        train_df=train_df,
        X_raw=X_emb,
        feature_engine=feat_engine,
        model_instance=model_instance,
        X_semantic=X_semantic
    )

    # 5. スコア計算と後処理
    y_true = train_df[CFG.TARGET_NAMES].values
    
    # 生の予測値でのスコア
    raw_score = competition_metric(y_true, oof_preds)
    print(f"Raw CV Score: {raw_score:.6f}")
    
    # 後処理（等式制約）を適用したスコア
    oof_df = pd.DataFrame(oof_preds, columns=CFG.TARGET_NAMES)
    processed_oof_df = post_process_biomass(oof_df)
    processed_score = competition_metric(y_true, processed_oof_df.values)
    
    print(f"Processed CV Score: {processed_score:.6f}")
    print(f"Improvement: {processed_score - raw_score:.6f}")

    # 6. 結果の保存
    processed_oof_df.to_csv("outputs/oof_preds.csv", index=False)
    print("Training Complete. OOF predictions saved to outputs/oof_preds.csv")

if __name__ == "__main__":
    main()