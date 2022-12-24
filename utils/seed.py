import os
import random
import numpy as np
import torch


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # reconstruction purpose
    torch.backends.cudnn.benchmark = False  # reconstruction purpose
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)
