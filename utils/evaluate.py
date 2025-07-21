import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.loss import combined_loss
from model.resnet_enc_dec import ResNet34UNet


def evaluate_model(
    model_path: str,
    test_loader: DataLoader,
    alpha: float = 0.85,
    beta:  float = 0.15,
    device: torch.device | None = None,
) -> tuple[float, float, float]:
    """
    Returns a tuple (avg_ssim, avg_mse, avg_combined_loss) on the test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet34UNet(pretrained=False)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    mse_fn = torch.nn.MSELoss()
    running_ssim     = 0.0
    running_mse      = 0.0
    running_combined = 0.0

    with torch.no_grad():
        for (
            _sino_fname,
            sino,
            img,
            sino_min, sino_max,
            img_min, img_max,
        ) in tqdm(test_loader, desc="Evaluating", ncols=100):
            sino = sino.to(device)
            img  = img.to(device)

            pred = model(sino)

            print("pred stats:", pred.min().item(), pred.max().item())
            print("gt stats:", img.min().item(), img.max().item())

            total_loss, ssim_val, mse_val = combined_loss(pred, img, alpha, beta, mse_fn)
            running_combined += total_loss.item()
            running_ssim     += ssim_val
            running_mse      += mse_val

    N = len(test_loader)
    avg_ssim     = running_ssim / N
    avg_mse      = running_mse / N
    avg_combined = running_combined / N

    print(
        f"✅ Test set → SSIM: {avg_ssim:.4f}  |  MSE: {avg_mse:.4f}  |  Combined Loss: {avg_combined:.4f}"
    )
    return avg_ssim, avg_mse, avg_combined

