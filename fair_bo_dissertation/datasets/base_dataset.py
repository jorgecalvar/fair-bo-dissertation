
from abc import ABC

from torch.utils.data import Dataset

class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


class BaseDataset(Dataset, ABC):

    def create_subset(self, indices):
        return SubsetDataset(self, indices)

