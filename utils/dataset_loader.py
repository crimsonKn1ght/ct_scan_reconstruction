import numpy as np
import os, re, h5py, glob, torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class PairedLoDoPaBDataset(Dataset):
    def __init__(self, folder, split='train'):
        self.samples = []
        gt_pattern = re.compile(rf'ground_truth_{split}_(\d+)\.hdf5')

        gt_map, obs_map = {}, {}
        for path in glob.glob(os.path.join(folder, '*.hdf5')):
            fname = os.path.basename(path)
            m = gt_pattern.match(fname)
            if m:
                idx = m.group(1)
                gt_map[idx] = path
            elif fname.startswith(f'observation_{split}_') and fname.endswith('.hdf5'):
                idx = re.findall(rf'observation_{split}_(\d+)\.hdf5', fname)
                if idx:
                    obs_map[idx[0]] = path

        common_indices = sorted(set(gt_map) & set(obs_map), key=lambda x: int(x))
        if not common_indices:
            raise RuntimeError(f"No matching pairs found for split='{split}' in {folder}")

        # Load samples with index
        for idx in common_indices:
            sino_path = obs_map[idx]
            img_path = gt_map[idx]

            with h5py.File(sino_path, 'r') as fs, h5py.File(img_path, 'r') as fi:
                n_samples = len(fs['data'])
                for i in range(n_samples):
                    unique_name = f"{os.path.splitext(os.path.basename(sino_path))[0]}_img{i}.npy"
                    self.samples.append((sino_path, img_path, i, unique_name))

        print(f"Found {len(self.samples)} total image pairs for split='{split}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sino_path, img_path, img_idx, unique_name = self.samples[idx]

        with h5py.File(sino_path, 'r') as fs, h5py.File(img_path, 'r') as fi:
            sino = fs['data'][img_idx]
            img  = fi['data'][img_idx]

        sino_min, sino_max = sino.min(), sino.max()
        img_min, img_max   = img.min(), img.max()

        sino_norm = (sino - sino_min) / (sino_max - sino_min + 1e-8)
        img_norm  = (img - img_min)   / (img_max  - img_min  + 1e-8)

        sino_norm = torch.tensor(sino_norm, dtype=torch.float32).unsqueeze(0)
        img_norm  = torch.tensor(img_norm,  dtype=torch.float32).unsqueeze(0)

        sino_norm = TF.resize(sino_norm, [128, 128], antialias=True)
        img_norm = TF.resize(img_norm, [128, 128], antialias=True)

        return (
            unique_name, sino_norm, img_norm,
            sino_min, sino_max, img_min, img_max,
        )
