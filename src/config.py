import torch
from pathlib import Path

class CFG:
    # パス設定
    BASE_PATH = Path("/kaggle/input/csiro-biomass") # ローカル環境に合わせて変更してください
    TRAIN_CSV = BASE_PATH / "train.csv"
    TEST_CSV = BASE_PATH / "test.csv"
    IMAGE_DIR = BASE_PATH / "train"
    
    # ターゲットと重み
    TARGET_NAMES = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    WEIGHTS = {
        'Dry_Green_g': 0.1,
        'Dry_Dead_g': 0.1,
        'Dry_Clover_g': 0.1,
        'GDM_g': 0.2,
        'Dry_Total_g': 0.5,
    }
    
    # 学習ハイパーパラメータ
    SEED = 42
    N_FOLDS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 画像設定
    IMG_SIZE = 518 # DINOv2の標準サイズ