import json
import os

import torch
from torch.utils.data import Dataset

from .. import images, paths
from ..logging import logger

CPUNKS_LABELS = os.path.join(paths.CPUNKS_ROOT, "data", "labels.json")


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

    def __init__(self, labels_file=None, test_size=0):
        self._images = self._load_punks()
        self._labels = self._load_labels(labels_file)
        split_idx = split_dataset(images.CPUNKS_SIZE, test_size)
        self.train_idx, self.test_idx = split_idx

    def _load_punk(self, i):
        return images.get_punk_tensor(i)

    def _load_punks(self):
        logger.info(f"loading punk images")

        punks = []
        for idx in range(images.CPUNKS_SIZE):
            if idx % 1000 == 0:
                logger.info(f"- image {idx}")
            p = self._load_punk(idx)
            punks.append(p)

        logger.info(f"punk images complete")
        return punks

    def _load_labels(self, labels_file):
        logger.info(f"loading punk labels: {labels_file}")

        # Look in artifacts dir for label name or fallback to CPUNKS_LABELS
        if labels_file:
            labels_file = os.path.join(paths.ARTIFACTS_DIR, labels_file)
            if not os.path.exists(labels_file):
                err_msg = f"ERROR: labels file not found {labels_file}"
                raise Exception(err_msg)
        else:
            labels_file = CPUNKS_LABELS
        all_labels = json.load(open(labels_file))

        # store keys for labels too
        self._label_keys = [k for k in all_labels["0"].keys()]

        labels = []
        for label in all_labels.values():
            t = torch.tensor([v for v in label.values()])
            labels.append(t)
        return labels

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = self._images[idx]
        labels = self._labels[idx]
        return image, labels


class FastCPunksDataset(CPunksDataset):
    """
    Same as CPunksDataset, but puts everything on the GPU
    """

    def __init__(self, device, *a, **kw):
        super(FastCPunksDataset, self).__init__(*a, **kw)
        # put data on GPU
        self.device = device
        self._images = torch.stack(self._images, 0).to(self.device)
        self._labels = torch.stack(self._labels, 0).to(self.device)
