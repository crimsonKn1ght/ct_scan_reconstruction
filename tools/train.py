# tools/train.py
from __future__ import annotations
import os, csv
from typing import Tuple
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_msssim import ssim

from tools.swin_enc_dec import SwinUNetStage1
from tools.resnet_enc_dec import RefinerUNet
from tools.dataset_loader import PairedLoDoPaBDataset


# --------------------
# Utility losses & metrics
# --------------------
def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1 - ssim(pred, target, data_range=1.0, size_average=True)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    return 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))


def combined_loss(
    pred_img: torch.Tensor,
    true_img: torch.Tensor,
    alpha: float,
    beta: float,
    mae_fn: nn.Module
):
    ssim_val = ssim(pred_img, true_img, data_range=1.0, size_average=True)
    loss_ssim = 1.0 - ssim_val
    mae_val = mae_fn(pred_img, true_img)
    img_loss = alpha * loss_ssim + beta * mae_val
    return img_loss, ssim_val.item(), mae_val.item()


def _remove_module_prefix(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


# --------------------
# Dataloaders
# --------------------
def build_loaders(
    root_dir: str,
    batch_size: int = 6,
    num_workers: int = 4,
    random_seed: int = 42,
    norm: str = "minmax",
    scale_val: float = 4096.0
):
    torch.manual_seed(random_seed)

    train_dataset = PairedLoDoPaBDataset(root_dir, split="train", norm=norm, scale_val=scale_val)
    val_dataset   = PairedLoDoPaBDataset(root_dir, split="validation", norm=norm, scale_val=scale_val)
    test_dataset  = PairedLoDoPaBDataset(root_dir, split="test", norm=norm, scale_val=scale_val)

    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **common_kwargs)
    val_loader   = DataLoader(val_dataset, shuffle=False, **common_kwargs)
    test_loader  = DataLoader(test_dataset, shuffle=False, **common_kwargs)

    return train_loader, val_loader, test_loader


# --------------------
# Training (Two Stage)
# --------------------
def train_two_stage(
    train_loader,
    val_loader,
    num_epochs=100,
    lr=3e-5,
    save_path="best_model.pth",
    log_path="training.csv",
    alpha=0.65,
    beta=0.35,
    weight_decay=1e-4,
    device=None,
    factor=0.3,
    patience=10,
    filter_type="ramp",
    lambda_refine=0.5,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Models ----
    stage1 = SwinUNetStage1(
        filter_type=filter_type,
        pretrained=True
    )
    refiner = RefinerUNet()

    if torch.cuda.device_count() > 1:
        stage1 = nn.DataParallel(stage1)
        refiner = nn.DataParallel(refiner)

    stage1 = stage1.to(device)
    refiner = refiner.to(device)

    # ---- Optimizers & schedulers ----
    opt1 = torch.optim.Adam(stage1.parameters(), lr=lr, weight_decay=weight_decay)
    opt2 = torch.optim.Adam(refiner.parameters(), lr=lr, weight_decay=weight_decay)

    sched1 = ReduceLROnPlateau(opt1, mode="min", factor=factor, patience=patience)
    sched2 = ReduceLROnPlateau(opt2, mode="min", factor=factor, patience=patience)

    mae_fn = nn.L1Loss()
    best_val = float("inf")
    start_epoch = 1

    # ---- Resume checkpoint ----
    if os.path.exists(save_path):
        print(f"📄 Resuming from checkpoint {save_path}")
        ckpt = torch.load(save_path, map_location=device)
        stage1.load_state_dict(_remove_module_prefix(ckpt["stage1"]))
        refiner.load_state_dict(_remove_module_prefix(ckpt["refiner"]))
        opt1.load_state_dict(ckpt["opt1"])
        opt2.load_state_dict(ckpt["opt2"])
        sched1.load_state_dict(ckpt["sched1"])
        sched2.load_state_dict(ckpt["sched2"])
        best_val = ckpt["best_val"]
        start_epoch = ckpt["epoch"] + 1

    # ---- CSV log header ----
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch",
            "train_loss1","train_loss2","train_comb",
            "train_ssim","train_mae","train_psnr",
            "val_loss1","val_loss2","val_comb",
            "val_ssim","val_mae","val_psnr"
        ])

    # ---- Epoch loop ----
    for epoch in range(start_epoch, num_epochs + 1):
        stage1.train()
        refiner.train()

        train_loss1 = train_loss2 = train_comb = 0.0
        train_ssim = train_mae = train_psnr = 0.0

        for _, sino, img, *_ in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            sino = sino.to(device)
            img  = img.to(device)

            pred1 = stage1(sino)
            pred2 = refiner(pred1)

            loss1, ssim1, mae1 = combined_loss(pred1, img, alpha, beta, mae_fn)
            loss2, ssim2, mae2 = combined_loss(pred2, img, alpha, beta, mae_fn)

            opt1.zero_grad(set_to_none=True)
            opt2.zero_grad(set_to_none=True)

            total_loss = loss1 + lambda_refine * loss2
            total_loss.backward()

            opt1.step()
            opt2.step()

            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_comb  += total_loss.item()
            train_ssim  += (ssim1 + ssim2) / 2
            train_mae   += (mae1  + mae2) / 2
            train_psnr  += (psnr(pred1, img).item() + psnr(pred2, img).item()) / 2

        n_train = len(train_loader)

        # ---- Validation ----
        stage1.eval()
        refiner.eval()

        val_loss1 = val_loss2 = val_comb = 0.0
        val_ssim = val_mae = val_psnr = 0.0

        with torch.no_grad():
            for _, sino, img, *_ in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                sino = sino.to(device)
                img  = img.to(device)

                pred1 = stage1(sino)
                pred2 = refiner(pred1)

                loss1, ssim1, mae1 = combined_loss(pred1, img, alpha, beta, mae_fn)
                loss2, ssim2, mae2 = combined_loss(pred2, img, alpha, beta, mae_fn)

                total_loss = loss1 + lambda_refine * loss2

                val_loss1 += loss1.item()
                val_loss2 += loss2.item()
                val_comb  += total_loss.item()
                val_ssim  += (ssim1 + ssim2) / 2
                val_mae   += (mae1  + mae2) / 2
                val_psnr  += (psnr(pred1, img).item() + psnr(pred2, img).item()) / 2

        val_loss1 /= len(val_loader)
        val_loss2 /= len(val_loader)
        val_comb  /= len(val_loader)
        val_ssim  /= len(val_loader)
        val_mae   /= len(val_loader)
        val_psnr  /= len(val_loader)

        print(
            f"Epoch {epoch} │ "
            f"Train SSIM {train_ssim/n_train:.4f} │ "
            f"Train PSNR {train_psnr/n_train:.2f} │ "
            f"Val SSIM {val_ssim:.4f} │ "
            f"Val PSNR {val_psnr:.2f}"
        )

        # ---- Save best ----
        if val_comb < best_val:
            best_val = val_comb
            torch.save({
                "epoch": epoch,
                "stage1": stage1.state_dict(),
                "refiner": refiner.state_dict(),
                "opt1": opt1.state_dict(),
                "opt2": opt2.state_dict(),
                "sched1": sched1.state_dict(),
                "sched2": sched2.state_dict(),
                "best_val": best_val
            }, save_path)
            print(f"💾 Saved best model (val loss {best_val:.6f})")

        sched1.step(val_comb)
        sched2.step(val_comb)

        # ---- CSV log ----
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                train_loss1/n_train, train_loss2/n_train, train_comb/n_train,
                train_ssim/n_train, train_mae/n_train, train_psnr/n_train,
                val_loss1, val_loss2, val_comb,
                val_ssim, val_mae, val_psnr
            ])

    return stage1, refiner, best_val


# --------------------
# Inference
# --------------------
def denormalize(pred_norm: np.ndarray, img_min: float, img_max: float) -> np.ndarray:
    return pred_norm * (img_max - img_min) + img_min


def inference_two_stage(stage1, refiner, test_loader, device=None, output_dir="predictions"):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1.eval()
    refiner.eval()

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            names, sino, img, sino_min, sino_max, img_min, img_max = batch
            sino = sino.to(device)

            pred1 = stage1(sino)
            pred2 = refiner(pred1)
            pred = pred2.cpu().numpy()

            img_min = img_min.numpy()
            img_max = img_max.numpy()

            for i in range(pred.shape[0]):
                p = pred[i, 0]
                p = torch.from_numpy(p).unsqueeze(0).unsqueeze(0)
                p = F.interpolate(p, size=(128, 128), mode="bicubic", align_corners=False)
                p = p.squeeze().numpy()
                out = denormalize(p, float(img_min[i]), float(img_max[i]))
                np.save(os.path.join(output_dir, f"pred_{names[i]}"), out)


# --------------------
# Evaluation
# --------------------
def evaluate_two_stage(
    model_path,
    test_loader,
    alpha=0.6,
    beta=0.4,
    filter_type="ramp",
    lambda_refine=0.5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage1 = SwinUNetStage1(filter_type=filter_type, pretrained=False)
    refiner = RefinerUNet()

    if torch.cuda.device_count() > 1:
        stage1 = nn.DataParallel(stage1)
        refiner = nn.DataParallel(refiner)

    stage1 = stage1.to(device)
    refiner = refiner.to(device)

    ckpt = torch.load(model_path, map_location=device)
    stage1.load_state_dict(_remove_module_prefix(ckpt["stage1"]))
    refiner.load_state_dict(_remove_module_prefix(ckpt["refiner"]))

    stage1.eval()
    refiner.eval()

    mae_fn = nn.L1Loss()

    ssim_vals, mae_vals, psnr_vals, comb_vals = [], [], [], []

    with torch.no_grad():
        for _, sino, img, *_ in tqdm(test_loader, desc="Evaluating"):
            sino = sino.to(device)
            img  = img.to(device)

            pred1 = stage1(sino)
            pred2 = refiner(pred1)

            loss1, ssim1, mae1 = combined_loss(pred1, img, alpha, beta, mae_fn)
            loss2, ssim2, mae2 = combined_loss(pred2, img, alpha, beta, mae_fn)

            total_loss = loss1 + lambda_refine * loss2

            ssim_vals.append((ssim1 + ssim2) / 2)
            mae_vals.append((mae1 + mae2) / 2)
            psnr_vals.append((psnr(pred1, img).item() + psnr(pred2, img).item()) / 2)
            comb_vals.append(total_loss.item())

    print(
        f"✅ Test → "
        f"SSIM: {np.mean(ssim_vals):.4f} | "
        f"PSNR: {np.mean(psnr_vals):.2f} dB | "
        f"MAE: {np.mean(mae_vals):.6f} | "
        f"Loss: {np.mean(comb_vals):.6f}"
    )

    return (
        np.mean(ssim_vals),
        np.mean(psnr_vals),
        np.mean(mae_vals),
        np.mean(comb_vals),
    )
