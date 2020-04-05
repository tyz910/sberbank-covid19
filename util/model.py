import torch
from torch.utils.data import Dataset


class CovidDataset(Dataset):
    def __init__(self, X, y=None):
        self.features = torch.from_numpy(X.values).float()
        if y is not None:
            self.targets = torch.from_numpy(y.values).float()
        else:
            self.targets = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        output = {
            "features": self.features[idx]
        }

        if self.targets is not None:
            output["targets"] = self.targets[idx]

        return output
