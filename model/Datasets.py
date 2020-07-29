import torch
import numpy as np

class CriteoDataset(torch.utils.data.Dataset):
    def __init__(self, X_index, X_value, labels):
        self.labels = labels
        self.X_index = np.array(X_index).reshape((-1, 26, 1))
        self.X_value = np.array(X_value)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        Xi_i = torch.tensor(self.X_index[index]).long()
        Xv_i = torch.tensor(self.X_value[index]).long()
        y = torch.tensor(self.labels[index]).float()

        return Xi_i, Xv_i, y
