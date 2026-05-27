import os
import random

import numpy as np


DEFAULT_SEED = 2026


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    """Set common pseudo-random seeds for reproducible experiments."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed
