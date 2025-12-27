"""
- 評価指標を計算する関数
- 後処理の関数
を実装
"""
import numpy as np
import pandas as pd
from src.config import CFG

def competition_metric(y_true, y_pred):
    """コンペ公式のWeighted R2を計算"""
    weights_arr = np.array([CFG.WEIGHTS[name] for name in CFG.TARGET_NAMES])
    
    # グローバルな平均値を計算（重み付き）
    y_weighted_mean = np.sum(np.mean(y_true, axis=0) * weights_arr)
    
    # 残差平方和と全平方和
    ss_res = np.sum(np.mean((y_true - y_pred)**2, axis=0) * weights_arr)
    ss_tot = np.sum(np.mean((y_true - y_weighted_mean)**2, axis=0) * weights_arr)
    
    return 1 - ss_res / ss_tot

def post_process_biomass(preds_df):
    """
    等式制約を適用して予測値を補正する
    Green + Clover = GDM
    GDM + Dead = Total
    """
    cols = CFG.TARGET_NAMES
    # 行列演算用の順序に合わせる (Green, Clover, Dead, GDM, Total)
    Y = preds_df[cols].values.T
    
    # 制約行列 C (等式を満たす場合に C @ Y = 0 となるように設計)
    # [1, 1, 0, -1, 0] -> Green + Clover - GDM = 0
    # [0, 0, 1, 1, -1] -> Dead + GDM - Total = 0
    C = np.array([
        [1, 1, 0, -1,  0], 
        [0, 0, 1,  1, -1]
    ])
    
    C_T = C.T
    inv_CCt = np.linalg.inv(C @ C_T)
    P = np.eye(5) - C_T @ inv_CCt @ C # 直交射影行列
    
    Y_reconciled = (P @ Y).T
    # 負の値をクリップ
    Y_reconciled = np.maximum(Y_reconciled, 0)
    
    return pd.DataFrame(Y_reconciled, columns=cols, index=preds_df.index)