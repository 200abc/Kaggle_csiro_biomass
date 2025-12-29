import os
import random
import numpy as np
import torch
import gc

def seeding(seed: int):
    """実験の再現性を確保するためにシード値を固定する"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Seed fixed: {seed}")

def flush_memory():
    """不要なメモリを解放する"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()