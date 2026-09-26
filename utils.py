import os
import random

import numpy as np
import torch

# Random seed for reproducibality.
def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Allow cuDNN to select fast kernels.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def relative_frobenius_error(original, reconstructed):

    numerator = torch.linalg.norm(original - reconstructed)
    denominator = torch.linalg.norm(original)

    if denominator == 0:
        raise ValueError("Original tensor has zero Frobenius norm.")

    return (numerator / denominator).item()