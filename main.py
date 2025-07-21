import numpy as np, os, random, re
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"          # Uncomment to use specific GPUs
import torch

from utils.train import train_model
from utils.evaluate import evaluate_model
from utils.inference import inference
from utils.build_loaders import build_loaders


print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version in torch:", torch.version.cuda)
print("cuDNN enabled:", torch.backends.cudnn.enabled)



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    set_seed(42)

    root_dir    =       "/app/dataset/"
    save_path   =       "/app/best_model.pth"
    log_path    =       "/app/results.csv"
    output_dir  =       "/app/saved_test_preds"

    train_loader, val_loader, test_loader = build_loaders(
        root_dir=root_dir,
        batch_size=64,
    )


    print("🟢 Starting training...")
    model, best_combined, best_ssim = train_model(
        train_loader,
        val_loader,
        num_epochs=124,
        save_path=save_path,
        log_path=log_path,
        alpha=0.85,
        beta=0.15,
        lr=1e-4,
        weight_decay=1e-4,
        factor=0.3,
        patience=10,
        # flg=True,
    )
    print(f"✅ Training completed. Best Val SSIM: {best_ssim:.4f}")

    print("🟢 Running inference on test set...")
    inference(model, test_loader, output_dir=output_dir)
    print(f"✅ Predictions saved to '{output_dir}'.")

    print("🟢 Evaluating saved model on test set...")
    evaluate_model(save_path, test_loader)
