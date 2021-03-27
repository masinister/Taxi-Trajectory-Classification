import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset
from utils import traj_to_tensor, parse_file
from tqdm import tqdm
import os

class TrajectoryDataset(Dataset):

    def __init__(self, dir, n_features, traj_len, transform=None):
        self.dir = dir
        self.n_features = n_features
        self.traj_len = traj_len
        self.data = []
        self.labels = []
        self.transform = transform
        for entry in tqdm(os.scandir(self.dir), desc = "Loading Files"):
            if entry.path.endswith(".csv") and entry.is_file():
                data, labels = parse_file(entry.path, traj_len = self.traj_len)
                self.data += [traj_to_tensor(traj, n_features = self.n_features, pad_len = self.traj_len) for traj in data]
                self.labels += [torch.tensor(l, dtype=torch.long, device = torch.device("cuda")) for l in labels]

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.data)
