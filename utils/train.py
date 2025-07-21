import numpy as np
from tqdm import tqdm
import os, csv, torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from utils.loss import combined_loss
from model.resnet_enc_dec import ResNet34UNet



print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"[{i}] {torch.cuda.get_device_name(i)}")


def train_model(
    train_loader: DataLoader,
    val_loader:   DataLoader,
    num_epochs:   int   = 100,
    lr:           float = 1e-5,
    save_path:    str   = "C:/Users/gouro/Desktop/best_model.pth",
    log_path:     str   = "C:/Users/gouro/Desktop/scheduler.csv",
    alpha:        float = 0.8,
    beta:         float = 0.2,
    weight_decay: float = 1e-5,
    device:       torch.device = None,
    factor:       float = 0.2,
    patience:     int   = 5,
    flg:          bool  = False,
):
    """
    Uses combined_loss = alpha*(1-SSIM) + beta*MSE.
    Returns (best_model, best_val_combined_loss).
    """

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet34UNet(pretrained=False)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
    else:
        print('Something\'s wrong')

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)

    try:
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
    except Exception as e:
        print(f"Encountered following error: {e}")
        start_epoch=1

    if start_epoch == 1:
        if flg == True:
            decoder_checkpoint = torch.load("/data/gourab/best_model - 500 epochs - comb loss - aemodel - img - img.pth", map_location="cpu")  # Your pretrained model file
            decoder_state = decoder_checkpoint["model_state"]

            model_state = model.state_dict()
            prefix = "module." if torch.cuda.device_count() > 1 else ""
            decoder_keys = ['dec4', 'dec3', 'dec2', 'dec1', 'final_up', 'out_conv']
            allowed_keys = tuple(f"{prefix}{k}" for k in decoder_keys)

            decoder_weights = {
                k: v for k, v in decoder_state.items() if k.startswith(allowed_keys)
            }

            model_state.update(decoder_weights)
            model.load_state_dict(model_state)

            print("✅ Loaded decoder weights from model.pth")
            print("Loaded keys:")
            for k in decoder_weights:
                print("  ", k)

            # for name, param in model.named_parameters():
            #     if any(name.startswith(k) for k in allowed_keys):
            #         param.requires_grad = False
            #         print(f"❄️  Frozen: {name}")



    mse_fn = torch.nn.MSELoss()          # pre‐instantiate MSE loss
    best_val_combined = float("inf")     # lower combined‐loss is better
    best_val_ssim     = -float("inf")
    
    if not os.path.exists(log_path):
        with open(log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Epoch",
                "Train_SSIM", "Train_MSE", "Train_Combined",
                "Val_SSIM", "Val_MSE", "Val_Combined",
                "Learning_Rate"
            ])


    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        running_ssim_train     = 0.0
        running_mse_train      = 0.0
        running_combined_train = 0.0

        for (
            _sino_fname,   # we don’t need the filename for training steps
            sino,
            img,
            sino_min, sino_max,
            img_min, img_max,
        ) in tqdm(train_loader, desc=f"Train {epoch}/{num_epochs}", ncols=100):
            sino = sino.to(device)
            img  = img.to(device)

            # Forward pass
            pred = model(sino)

            # Compute combined loss (tensor), plus scalar SSIM/MSE for logging
            total_loss, ssim_val, mse_val = combined_loss(pred, img, alpha, beta, mse_fn)

            # Backprop & optimizer step
            optimizer.zero_grad()
            total_loss.backward()    # <-- now total_loss is a tensor, so .backward() works
            optimizer.step()

            # Accumulate for epoch statistics
            running_combined_train += total_loss.item()
            running_ssim_train     += ssim_val
            running_mse_train      += mse_val

        # Averages over training batches
        num_train_batches   = len(train_loader)
        avg_combined_train  = running_combined_train / num_train_batches
        avg_ssim_train      = running_ssim_train / num_train_batches
        avg_mse_train       = running_mse_train / num_train_batches


        # ──────────────────────────────────────────────────────
        # Validation phase
        model.eval()
        running_ssim_val     = 0.0
        running_mse_val      = 0.0
        running_combined_val = 0.0

        with torch.no_grad():
            for (
                _sino_fname,
                sino,
                img,
                sino_min, sino_max,
                img_min, img_max,
            ) in val_loader:
                sino = sino.to(device)
                img  = img.to(device)

                pred = model(sino)
                total_loss, ssim_val, mse_val = combined_loss(pred, img, alpha, beta, mse_fn)

                running_combined_val += total_loss.item()
                running_ssim_val     += ssim_val
                running_mse_val      += mse_val

        num_val_batches  = len(val_loader)
        avg_combined_val = running_combined_val / num_val_batches
        avg_ssim_val     = running_ssim_val / num_val_batches
        avg_mse_val      = running_mse_val / num_val_batches
        current_lr       = optimizer.param_groups[0]['lr']

        # ──────────────────────────────────────────────────────
        # Print a concise summary for this epoch
        print(
            f"→ Epoch {epoch:02d} │ "
            f"Train [SSIM: {avg_ssim_train:.4f} | MSE: {avg_mse_train:.4f} | Combined: {avg_combined_train:.4f}]  │  "
            f"Val   [SSIM: {avg_ssim_val:.4f} | MSE: {avg_mse_val:.4f} | Combined: {avg_combined_val:.4f}]  │  "
            f"Current LR: {current_lr:.2e}"
        )

        # Checkpoint on lowest validation combined‐loss
        if avg_combined_val < best_val_combined:
            best_val_combined = avg_combined_val
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_combined_loss": best_val_combined,
                "val_ssim_at_checkpoint": avg_ssim_val
            }, save_path)
            print(f"    ↑ Saved new best model (Val Combined Loss {best_val_combined:.4f})")
            
        if avg_ssim_val > best_val_ssim:
            best_val_ssim = avg_ssim_val
        
        with open(log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch,
                round(avg_ssim_train, 6), round(avg_mse_train, 6), round(avg_combined_train, 6),
                round(avg_ssim_val, 6), round(avg_mse_val, 6), round(avg_combined_val, 6),
                round(current_lr, 8)
            ])

        scheduler.step(avg_combined_val)
        
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    return model, best_val_combined, best_val_ssim

