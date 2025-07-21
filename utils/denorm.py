import numpy as np

def denormalize(pred_norm: np.ndarray, img_min: float, img_max: float) -> np.ndarray:
    """Reverses min-max normalization: pred_denorm = pred_norm * (max - min) + min."""
    return pred_norm * (img_max - img_min) + img_min