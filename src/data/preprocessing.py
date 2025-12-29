"""
このファイルは、**「生のデータ（CSVや画像）をモデルが読み込める形に整える」**という重要な役割を担います。
ノートブックにあった複雑なピボット処理や、高解像度画像を扱うためのタイル分割ロジックをここに集約します。
"""

import cv2
import numpy as np
import pandas as pd
from src.config import CFG

def pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    ターゲットが縦に並んでいる(Long format)データを、
    モデルが扱いやすい横並び(Wide format)に変換する。
    """
    if 'target' in df.columns:
        # 学習用データの処理
        df_pt = pd.pivot_table(
            df, 
            values='target', 
            index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'], 
            columns='target_name', 
            aggfunc='mean'
        ).reset_index()
    else:
        # テスト用データの処理（ターゲット列がない場合）
        # ダミーのターゲット列を作成してピボットを回す
        temp_df = df.copy()
        temp_df['target'] = 0
        df_pt = pd.pivot_table(
            temp_df, 
            values='target', 
            index='image_path', 
            columns='target_name', 
            aggfunc='mean'
        ).reset_index()
        # 不要なダミー列を削除したければここで処理
    
    return df_pt

def melt_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    モデルの予測結果(Wide format)を、
    提出形式(Long format)のsample_id付きデータに変換する。
    """
    melted = df.melt(
        id_vars='image_path',
        value_vars=CFG.TARGET_NAMES,
        var_name='target_name',
        value_name='target'
    )
    
    # 公式の提出形式に必要な sample_id (image_id__target_name) を作成
    # image_path からファイル名だけを抽出して結合
    melted['sample_id'] = (
        melted['image_path']
        .str.replace(r'^.*/', '', regex=True)
        .str.replace('.jpg', '', regex=False)
        + '__' + melted['target_name']
    )
    
    return melted[['sample_id', 'image_path', 'target_name', 'target']]

def split_image(image: np.ndarray, patch_size: int = 520, overlap: int = 16):
    """
    高解像度画像をパッチに分割する。
    画像が patch_size で割り切れない場合は、反射（reflect）モードでパディングを行う。
    """
    h, w, c = image.shape
    stride = patch_size - overlap
    patches, coords = [], []
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1, x1, y2, x2 = y, x, y + patch_size, x + patch_size
            patch = image[y1:y2, x1:x2, :]
            
            # パッチサイズが足りない場合のパディング処理
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                pad_h = patch_size - patch.shape[0]
                pad_w = patch_size - patch.shape[1]
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            
            patches.append(patch)
            coords.append((y1, x1, y2, x2))
            
    return patches, coords

def preprocess_image_path(df: pd.DataFrame, base_path: str) -> pd.DataFrame:
    """
    CSV内の相対パスをフルパスに変換する
    """
    df = df.copy()
    df['image_path'] = df['image_path'].apply(lambda p: str(base_path / p))
    return df