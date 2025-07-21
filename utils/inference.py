import os
import torch
import numpy as np
from utils.denorm import denormalize


def inference(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device | None = None,
    output_dir: str = "predictions"
):
    """
    Runs inference on normalized sinograms and saves denormalized predictions.
    Saves each prediction as 'pred_<original_filename>.npy' in output_dir.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch
            sino_fname_list, sino, img, sino_min, sino_max, img_min, img_max = batch

            sino     = sino.to(device)               # (B,1,H,W)
            pred     = model(sino).cpu().numpy()     # (B,1,H,W)
            img_min  = img_min.numpy()
            img_max  = img_max.numpy()

            for j in range(pred.shape[0]):
                # (1) Get single prediction, remove channel dim
                pred_norm = pred[j, 0]               # shape (H,W)

                # (2) Denormalize
                pred_denorm = denormalize(pred_norm, img_min[j], img_max[j])

                # (3) Save
                original_fname = sino_fname_list[j]
                pred_fname     = f"pred_{original_fname}"
                out_path       = os.path.join(output_dir, pred_fname)

                np.save(out_path, pred_denorm)

                print(f"✅ Saved: {out_path}")