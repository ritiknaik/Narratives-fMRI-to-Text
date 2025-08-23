import torch
import torch.nn as nn
import numpy as np
import os
import torch.optim as optim
import torch.nn.functional as F
import h5py

class FMRI2Embedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FMRI2Embedding, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
        self.dataset_path = "../dataset/"

        self.dataset_paths = self.prepare_file_list(self.dataset_path)

    def forward(self, x):
        out = self.model(x)
        return nn.functional.normalize(out, dim=-1)
    
    def clip_contrastive_loss(self, pred, target, temperature=0.07):
        logits = torch.matmul(pred, target.T) / temperature 
        labels = torch.arange(len(pred)).to(pred.device)    

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2
    
    def prepare_file_list(self, dataset_dir):
        all_files = os.listdir(dataset_dir)

        file_pairs = []
        base_names = set(f.split('.')[0] for f in all_files)

        for name in base_names:
            h5f_path = os.path.join(dataset_dir, f"{name}.hf5")
            tg_path = os.path.join(dataset_dir, f"{name}.TextGrid")

            if os.path.exists(h5f_path) and os.path.exists(tg_path):
                try:
                    with h5py.File(h5f_path, 'r') as f:
                        data = f['data'][:]
                        if np.isnan(data).any() or np.isinf(data).any():
                            print(f"Skipping {name} due to NaNs/Infs in {h5f_path}")
                            continue
                except Exception as e:
                    print(f"Error loading {h5f_path}: {e}")
                    continue

                file_pairs.append([h5f_path, tg_path])

        # Print selected valid pairs
        for h5f, tg in file_pairs:
            print(f"H5F: {h5f}  |  TextGrid: {tg}")

        return file_pairs
    
class SpatialFMRI2Embedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.2),

            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
        self.dataset_path = "../dataset/"

        self.dataset_paths = self.prepare_file_list(self.dataset_path)

    def forward(self, x):
        return self.model(x)
    
    def clip_contrastive_loss(self, pred, target, temperature=0.07):
        logits = torch.matmul(pred, target.T) / temperature  # shape (B, B)
        labels = torch.arange(len(pred)).to(pred.device)     # shape (B,)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2
    
    def prepare_file_list(self, dataset_dir):
        all_files = os.listdir(dataset_dir)

        file_pairs = []
        base_names = set(f.split('.')[0] for f in all_files)

        for name in base_names:
            h5f_path = os.path.join(dataset_dir, f"{name}.hf5")
            tg_path = os.path.join(dataset_dir, f"{name}.TextGrid")

            if os.path.exists(h5f_path) and os.path.exists(tg_path):
                try:
                    with h5py.File(h5f_path, 'r') as f:
                        data = f['data'][:]
                        if np.isnan(data).any() or np.isinf(data).any():
                            print(f"Skipping {name} due to NaNs/Infs in {h5f_path}")
                            continue
                except Exception as e:
                    print(f"Error loading {h5f_path}: {e}")
                    continue

                file_pairs.append([h5f_path, tg_path])

        for h5f, tg in file_pairs:
            print(f"✅ H5F: {h5f}  |  TextGrid: {tg}")

        return file_pairs