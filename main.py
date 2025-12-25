from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"
import argparse, torch, numpy as np

from tools.train import build_loaders, train_two_stage, inference_two_stage, evaluate_two_stage


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, default='/data/pradipta/Reconstruction/LoDoPaB/3384092/', help="LoDoPaB folder with HDF5 files")
    p.add_argument("--save_path", type=str, default="/data/gourab/model_128_1.pth")
    p.add_argument("--log_path", type=str, default="/data/gourab/training_128_1.csv")
    p.add_argument("--output_dir", type=str, default="/data/gourab/predictions_128_1")
    p.add_argument("--batch_size", type=int, default=15)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=0.65)
    p.add_argument("--beta", type=float, default=0.35)
    p.add_argument("--factor", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--norm", type=str, default="minmax", choices=["minmax","scale"])
    p.add_argument("--scale_val", type=float, default=4096.0)
    return p.parse_args()


if __name__ == "__main__":
    set_seed(42)
    args = parse_args()

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA (torch):", torch.version.cuda)
    print("cuDNN enabled:", torch.backends.cudnn.enabled)

    train_loader, val_loader, test_loader = build_loaders(
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        norm=args.norm,
        scale_val=args.scale_val,
    )

    # ---------------------------
    # Training (resume or scratch)
    # ---------------------------
    if os.path.exists(args.save_path):
        print(f"🟢 Found checkpoint at {args.save_path}, resuming training...")
    else:
        print("🟢 No checkpoint found, starting training from scratch...")

    stage1, refiner, best_val = train_two_stage(
        train_loader, val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_path,
        log_path=args.log_path,
        alpha=args.alpha,
        beta=args.beta,
        weight_decay=args.weight_decay,
        factor=args.factor,
        patience=args.patience,
        filter_type="ramp",
    )
    print(f"✅ Training finished. Best Val Loss: {best_val:.6f}")

    # ---------------------------
    # Inference + Evaluation
    # ---------------------------
    print("🟢 Running inference...")
    inference_two_stage(stage1, refiner, test_loader, output_dir=args.output_dir)
    print(f"✅ Predictions saved to '{args.output_dir}'.")

    print("🟢 Evaluating saved models on test set...")
    evaluate_two_stage(args.save_path, test_loader)
