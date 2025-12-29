"""
このファイルでは、SigLIP や DINOv2 といった強力な事前学習済みモデルを使用して、画像を「意味のある数値（ベクトル）」に変換します。
パッチ分割して平均をとるロジックや、テキスト（「緑の草」など）との類似度を測るセマンティック特徴量の生成もここに集約します。
"""

import torch
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoImageProcessor, AutoModel, AutoTokenizer, SiglipProcessor
from src.config import CFG
from src.data.preprocessing import split_image

def flush_memory():
    """GPUメモリを解放するユーティリティ"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def get_model(model_path: str, device: str = 'cpu'):
    """モデルとプロセッサをロードする"""
    
    # 1. パスをOSが認識できる「絶対パス」に強制変換する
    # これにより、transformersがHub(URL)ではなくLocal(Folder)だと確信します
    target_path = Path(model_path).absolute()
    
    if not target_path.exists():
        raise FileNotFoundError(f"Model path not found: {target_path}")
    
    model_path_str = str(target_path)
    print(f"Loading model from: {model_path_str}")

    # 2. ロード実行
    model = AutoModel.from_pretrained(
        model_path_str, 
        local_files_only=True,
        trust_remote_code=True
    )
    
    try:
        # SigLIP専用のプロセッサを優先的に試す
        processor = SiglipProcessor.from_pretrained(model_path_str, local_files_only=True)
    except Exception:
        processor = AutoImageProcessor.from_pretrained(model_path_str, local_files_only=True)
        
    return model.to(device), processor
@torch.no_grad()
def compute_embeddings(df: pd.DataFrame, model_path: str, patch_size: int = 520) -> pd.DataFrame:
    """
    画像からベクトル特徴量を抽出する。
    1. 画像をパッチに分割
    2. 各パッチをモデルに通す
    3. パッチごとの特徴量を平均して1つのベクトルにする
    """
    device = CFG.DEVICE
    model, processor = get_model(model_path, device)
    
    IMAGE_PATHS, EMBEDDINGS = [], []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting features from {model_path}"):
        img_path = row['image_path']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # パッチ分割
        patches, _ = split_image(img, patch_size=patch_size)
        images = [Image.fromarray(p) for p in patches]
        
        # モデルへの入力準備
        inputs = processor(images=images, return_tensors="pt").to(device)
        
        # 特徴量抽出
        if 'siglip' in model_path.lower():
            features = model.get_image_features(**inputs)
        elif 'dino' in model_path.lower():
            features = model(**inputs).pooler_output
        else:
            # 一般的なモデル
            features = model(**inputs).last_hidden_state[:, 0, :]
            
        # 全パッチの平均をとる
        embeds = features.mean(dim=0).cpu().numpy()
        
        EMBEDDINGS.append(embeds)
        IMAGE_PATHS.append(img_path)
        
    embeddings = np.stack(EMBEDDINGS, axis=0)
    n_features = embeddings.shape[1]
    
    # DataFrame形式に整形
    emb_columns = [f"emb{i+1}" for i in range(n_features)]
    emb_df = pd.DataFrame(embeddings, columns=emb_columns)
    emb_df['image_path'] = IMAGE_PATHS
    
    flush_memory()
    return emb_df

@torch.no_grad()
def generate_semantic_features(image_embeddings: np.ndarray, model_path: str) -> np.ndarray:
    """
    SigLIPのテキスト対照学習を利用して、特定の概念（緑度、クローバー等）への類似度を計算する。
    """
    device = CFG.DEVICE
    model = AutoModel.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 農業的に意味のある概念の定義
    concept_groups = {
        "bare": ["bare soil", "dirt ground", "sparse vegetation"],
        "dense": ["dense tall pasture", "thick grassy volume", "high biomass"],
        "green": ["lush green vibrant pasture", "fresh growth"],
        "dead": ["dry brown dead grass", "yellow straw"],
        "clover": ["white clover", "trifolium repens", "broadleaf legume"]
    }
    
    concept_vectors = {}
    for name, prompts in concept_groups.items():
        inputs = tokenizer(prompts, padding="max_length", return_tensors="pt").to(device)
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        concept_vectors[name] = emb.mean(dim=0, keepdim=True)
        
    # 画像特徴量を正規化
    img_tensor = torch.tensor(image_embeddings, dtype=torch.float32).to(device)
    img_tensor = img_tensor / img_tensor.norm(p=2, dim=-1, keepdim=True)
    
    scores = {}
    for name, vec in concept_vectors.items():
        # コサイン類似度を計算
        scores[name] = torch.matmul(img_tensor, vec.T).cpu().numpy().flatten()
    
    df_scores = pd.DataFrame(scores)
    
    # 独自の比率特徴量の生成
    df_scores['ratio_greenness'] = df_scores['green'] / (df_scores['green'] + df_scores['dead'] + 1e-6)
    
    return df_scores.values