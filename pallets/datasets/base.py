import torch
from torch.utils.data import Dataset

from .. import images


def split_dataset(ds_size, test_size=0):
    """
    Splits a list of data into two randomized sets of indexes.
    """
    # shuffle indexes
    data_indices = torch.randperm(ds_size).tolist()

    a_idx = data_indices[test_size:]
    b_idx = data_indices[:test_size]

    return a_idx, b_idx


class CPunksDataset(Dataset):
    """
    Pytorch dataset that provides all images from cpunks-10k as torch tensors
    """
    SIZE = 10000

    def __init__(self, test_size=0):
        self._images = [images.get_punk_tensor(i) for i in range(self.SIZE)]
        split_idx = split_dataset(self.SIZE, test_size)
        self.train_idx, self.test_idx = split_idx

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = self._images[idx]
        return image
