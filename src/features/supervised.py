"""
1024次元（SigLIPの場合など）の巨大なベクトルをそのままLGBMに入れると、データ数（画像数）に対して次元が多すぎて過学習しやすくなります。
これをPCA（主成分分析）やPLS（部分的最小二乗回帰）で、情報の密度を高めて凝縮します。
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.mixture import GaussianMixture

class SupervisedEmbeddingEngine(BaseEstimator, TransformerMixin):
    """
    AIベクトルを統計的に圧縮・加工するエンジン
    - PCA: 教師なしでの次元圧縮
    - PLS: ターゲット情報を使った次元圧縮（非常に強力）
    - GMM: データの「群れ」としての特徴を抽出
    """
    def __init__(self, n_pca=0.80, n_pls=8, n_gmm=6, random_state=42):
        self.n_pca = n_pca
        self.n_pls = n_pls
        self.n_gmm = n_gmm
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca, random_state=random_state)
        self.pls = PLSRegression(n_components=n_pls, scale=False)
        self.gmm = GaussianMixture(n_components=n_gmm, covariance_type='diag', random_state=random_state)
        self.pls_fitted_ = False

    def fit(self, X, y=None, X_semantic=None):
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        self.gmm.fit(X_scaled)
        
        if y is not None:
            # PLSはターゲット(y)を見て、予測に役立つ方向に軸を引く
            y_clean = y.values if hasattr(y, 'values') else y
            self.pls.fit(X_scaled, y_clean)
            self.pls_fitted_ = True
        return self

    def transform(self, X, X_semantic=None):
        X_scaled = self.scaler.transform(X)
        features = []
        
        # PCA特徴量
        features.append(self.pca.transform(X_scaled))
        
        # PLS特徴量 (ターゲットとの相関が強い軸)
        if self.pls_fitted_:
            f_pls, _ = self.pls.transform(X_scaled), None # PLSの戻り値はタプル
            features.append(f_pls)
            
        # GMM特徴量 (どのクラスタに属するかという確率)
        features.append(self.gmm.predict_proba(X_scaled))
        
        # セマンティック特徴量があれば結合
        if X_semantic is not None:
            # 簡単に正規化して結合
            sem_norm = (X_semantic - np.mean(X_semantic, axis=0)) / (np.std(X_semantic, axis=0) + 1e-6)
            features.append(sem_norm)
            
        return np.hstack(features)