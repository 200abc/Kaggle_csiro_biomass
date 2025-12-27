"""
5つのターゲットを効率的に学習するためのラッパーです。
ターゲットごとに値の範囲が異なるため、最大値で割って「0〜1」の範囲にしてから学習させる工夫（Target Transformation）などを盛り込みます。
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from src.config import CFG
from src.utils.metrics import competition_metric

class MultiTargetRegressor:
    """
    5つのターゲットそれぞれに対して回帰モデルを学習・推論するクラス
    """
    def __init__(self, base_model, target_transform='max'):
        self.base_model = base_model
        self.target_transform = target_transform
        self.models = {}
        # ターゲットごとの最大値（元のノートブックの値を流用）
        self.target_max = {
            "Dry_Clover_g": 71.7865,
            "Dry_Dead_g": 83.8407,
            "Dry_Green_g": 157.9836,
            "Dry_Total_g": 185.70,
            "GDM_g": 157.9836,
        }

    def _transform_target(self, y, target_name):
        if self.target_transform == 'max':
            return y / self.target_max[target_name]
        elif self.target_transform == 'log':
            return np.log1p(y)
        return y

    def _inverse_transform_target(self, y_pred, target_name):
        if self.target_transform == 'max':
            return y_pred * self.target_max[target_name]
        elif self.target_transform == 'log':
            return np.expm1(y_pred)
        return y_pred

    def fit(self, X, y_df):
        for target in CFG.TARGET_NAMES:
            model = deepcopy(self.base_model)
            y_scaled = self._transform_target(y_df[target].values, target)
            model.fit(X, y_scaled)
            self.models[target] = model

    def predict(self, X):
        preds = {}
        for target in CFG.TARGET_NAMES:
            raw_pred = self.models[target].predict(X)
            # 逆変換して元の単位（グラム）に戻す
            preds[target] = self._inverse_transform_target(raw_pred, target)
        return pd.DataFrame(preds)

def run_tabular_cv(train_df, X_raw, feature_engine, model_instance, X_semantic=None):
    """
    交差検証(CV)を実行し、OOF予測値を返す
    """
    folds = train_df['fold'].unique()
    oof_preds = np.zeros((len(train_df), len(CFG.TARGET_NAMES)))
    
    for fold in sorted(folds):
        train_idx = train_df[train_df['fold'] != fold].index
        valid_idx = train_df[train_df['fold'] == fold].index
        
        # 特徴量エンジニアリング（このフォルダのデータだけでFitさせるのが重要）
        X_train_raw = X_raw[train_idx]
        y_train = train_df.loc[train_idx, CFG.TARGET_NAMES]
        sem_train = X_semantic[train_idx] if X_semantic is not None else None
        
        engine = deepcopy(feature_engine)
        engine.fit(X_train_raw, y=y_train, X_semantic=sem_train)
        
        # 変換
        X_train = engine.transform(X_train_raw, X_semantic=sem_train)
        X_valid = engine.transform(X_raw[valid_idx], X_semantic=(X_semantic[valid_idx] if X_semantic is not None else None))
        
        # モデル学習
        multi_model = MultiTargetRegressor(model_instance)
        multi_model.fit(X_train, y_train)
        
        # 予測
        fold_preds = multi_model.predict(X_valid)
        oof_preds[valid_idx] = fold_preds.values
        
    return oof_preds