import os, re, glob, h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class PairedLoDoPaBDataset(Dataset):
    """
    Loads paired HDF5: observation_* (sinograms) and ground_truth_* (CT images).
    - Resizes to 128x128 to match the model.
    - Normalization: 'minmax' (default) or 'scale' (divide by `scale_val`).
    - Caches HDF5 file handles per worker to avoid reopen overhead.
    """
    def __init__(self, folder, split='train', norm: str = 'scale', scale_val: float = 4096.0):
        self.folder = folder
        self.split = split
        self.norm = norm
        self.scale_val = float(scale_val)
        self.samples = []
        self._h5_cache = {}

        gt_pattern = re.compile(rf'ground_truth_{split}_(\d+)\.hdf5')
        gt_map, obs_map = {}, {}

        for path in glob.glob(os.path.join(folder, '*.hdf5')):
            fname = os.path.basename(path)
            m = gt_pattern.match(fname)
            if m:
                gt_map[m.group(1)] = path
                continue
            if fname.startswith(f'observation_{split}_') and fname.endswith('.hdf5'):
                m2 = re.findall(rf'observation_{split}_(\d+)\.hdf5', fname)
                if m2:
                    obs_map[m2[0]] = path

        common = sorted(set(gt_map) & set(obs_map), key=lambda x: int(x))
        if not common:
            raise RuntimeError(f"No matching pairs found for split='{split}' in {folder}")

        for idx in common:
            sino_path = obs_map[idx]
            img_path  = gt_map[idx]
            with h5py.File(sino_path, 'r') as fs:
                n_samples = len(fs['data'])
            for i in range(n_samples):
                unique_name = f"{os.path.splitext(os.path.basename(sino_path))[0]}_img{i}.npy"
                self.samples.append((sino_path, img_path, i, unique_name))

        print(f"Found {len(self.samples)} total image pairs for split='{split}'")

    def __len__(self):
        return len(self.samples)

    def _get_file(self, path: str) -> h5py.File:
        f = self._h5_cache.get(path)
        if f is None or not f.__bool__():
            f = h5py.File(path, 'r')
            self._h5_cache[path] = f
        return f

    def __getitem__(self, idx):
        sino_path, img_path, img_idx, unique_name = self.samples[idx]

        fs = self._get_file(sino_path)
        fi = self._get_file(img_path)

        sino = fs['data'][img_idx]
        img  = fi['data'][img_idx]

        if self.norm == 'scale':
            sino_min, sino_max = 0.0, float(self.scale_val)
            img_min,  img_max  = 0.0, float(self.scale_val)
            sino_norm = np.clip(sino / self.scale_val, 0.0, 1.0)
            img_norm  = np.clip(img  / self.scale_val, 0.0, 1.0)
        else:
            sino_min, sino_max = float(sino.min()), float(sino.max())
            img_min,  img_max  = float(img.min()),  float(img.max())
            sino_norm = (sino - sino_min) / (sino_max - sino_min + 1e-8)
            img_norm  = (img  - img_min)  / (img_max  - img_min  + 1e-8)

        sino_t = torch.as_tensor(sino_norm, dtype=torch.float32).unsqueeze(0)
        img_t  = torch.as_tensor(img_norm,  dtype=torch.float32).unsqueeze(0)

        # 🔹 Resize to 128×128
        # resize to 362x362
        sino_t = TF.resize(sino_t, [362, 362], antialias=True)
        img_t  = TF.resize(img_t,  [362, 362], antialias=True)


        sino_min = torch.tensor(sino_min, dtype=torch.float32)
        sino_max = torch.tensor(sino_max, dtype=torch.float32)
        img_min  = torch.tensor(img_min,  dtype=torch.float32)
        img_max  = torch.tensor(img_max,  dtype=torch.float32)

        return (unique_name, sino_t, img_t, sino_min, sino_max, img_min, img_max)

    def __del__(self):
        for f in self._h5_cache.values():
            try:
                f.close()
            except Exception:
                pass
