# End-to-End CT Reconstruction with Swin Transformer

This repository implements an **end-to-end CT image reconstruction pipeline** that maps **sinograms → reconstructed CT images** using deep learning.

The system follows a **two-stage architecture**:

1. **Stage 1**: Swin Transformer–based encoder–decoder for global reconstruction
2. **Stage 2**: Lightweight convolutional UNet refiner for local detail enhancement

The pipeline is designed for **LoDoPaB-style datasets** and supports **multi-GPU training**.

---

## Features

- End-to-end sinogram → image reconstruction
- Swin Transformer encoder with convolutional decoder
- Physics-inspired sinogram-domain filtering (Ramp / Hann)
- Two-stage reconstruction with refinement
- Metrics: **SSIM, PSNR, MAE**
- Multi-GPU support via `DataParallel`
- Robust to different `timm` Swin output formats

---

## Project Structure

```
.
├── main.py
├── tools/
│   ├── train.py
│   ├── swin_enc_dec.py
│   ├── resnet_enc_dec.py
│   ├── dataset_loader.py
│   └── filters.py
```

---

## Dataset

The code expects a **LoDoPaB-style directory** containing paired HDF5 files:

```
root_dir/
├── observation_train_*.hdf5
├── ground_truth_train_*.hdf5
├── observation_validation_*.hdf5
├── ground_truth_validation_*.hdf5
├── observation_test_*.hdf5
└── ground_truth_test_*.hdf5
```

Each HDF5 file must contain a dataset named `data`.

---

## Installation

```
pip install torch torchvision timm pytorch-msssim h5py tqdm numpy
```

Tested with:
- PyTorch ≥ 2.0
- CUDA 11.8
- `timm` Swin Transformer models

---

## Training

```
python main.py \
  --root_dir /path/to/LoDoPaB \
  --save_path /path/to/model.pth \
  --log_path /path/to/training.csv \
  --output_dir /path/to/predictions \
  --epochs 100 \
  --batch_size 1 \
  --lr 3e-5
```

**Important:** If you change the model architecture, delete old checkpoints to avoid shape mismatches.

---

## Outputs

- **Model checkpoints**: saved to `--save_path`
- **Training logs**: CSV file with loss, SSIM, PSNR, MAE
- **Predictions**: NumPy `.npy` files (denormalized CT images)

---

## Evaluation Metrics

- **SSIM** — Structural Similarity Index
- **PSNR** — Peak Signal-to-Noise Ratio
- **MAE** — Mean Absolute Error

Metrics are computed during training, validation, and testing.

---

## Notes

- Inputs are padded to `384×384` for Swin compatibility and cropped back to original resolution.
- Stage 1 captures global structure; Stage 2 improves local texture.
- Design prioritizes **stability and reproducibility**.

---

## License

This project is intended for **research and educational use**.

