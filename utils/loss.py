import torch
from pytorch_msssim import ssim



def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta:  float,
    mse_fn: torch.nn.Module = None
) -> tuple[torch.Tensor, float, float]:


    # 1) Compute SSIM as a tensor (single‐scale SSIM from pytorch_msssim)
    ssim_map = ssim(pred, target, data_range=1.0)  # returns a scalar tensor
    loss_ssim_tensor = 1.0 - ssim_map               # also a tensor

    # 2) Compute MSE as a tensor
    if mse_fn is None:
        mse_fn = torch.nn.MSELoss()
    mse_tensor = mse_fn(pred, target)  # scalar tensor

    # 3) Combine them into a single Tensor
    total_loss = alpha * loss_ssim_tensor + beta * mse_tensor

    # 4) Extract Python floats for logging
    ssim_val = ssim_map.item()       # SSIM value ∈ [0,1]
    mse_val  = mse_tensor.item()

    return total_loss, ssim_val, mse_val

