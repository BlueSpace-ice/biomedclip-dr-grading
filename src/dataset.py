# -*- coding: utf-8 -*-
"""
Image-level H5 dataset.

* Training split -> (image_tensor, label, token_ids)
    token_ids come from findings_en (the semantic anchor). diagnosis_en is NOT
    used (it paraphrases the label and would weaken the contrastive signal).
* Eval splits (val / *_test) -> (image_tensor, label)
    text is never loaded -> matches the paper's unimodal inference.

h5py handles are opened lazily inside each worker process (fork/spawn safe).
"""
import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import config as C


def build_transforms(image_mean, image_std, train):
    if train:
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(image_mean, image_std),
        ])
    return T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(image_mean, image_std),
    ])


class FFADataset(Dataset):
    def __init__(self, h5_path, image_mean, image_std, train, tokenizer=None,
                 text_field=C.TEXT_FIELD, max_text_len=C.MAX_TEXT_LEN):
        self.path = h5_path
        self.train = train
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.max_text_len = max_text_len
        self.tf = build_transforms(image_mean, image_std, train)
        self._h5 = None
        # capture key order once (sorted numerically) without holding the handle
        with h5py.File(h5_path, "r") as f:
            self.keys = sorted(f["images"].keys(), key=lambda k: int(k))
            self.has_text = text_field in f

    def _f(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        f = self._f()
        k = self.keys[idx]
        arr = f["images"][k][()]                       # (224,224,3) uint8
        img = self.tf(Image.fromarray(arr))
        label = int(f["label"][k][()])

        if self.train and self.tokenizer is not None and self.has_text:
            txt = bytes(f[self.text_field][k][()]).decode("utf-8", errors="replace")
            toks = self.tokenizer([txt])               # (1, context_len) LongTensor
            return img, label, toks[0]
        return img, label


def make_loader(split, image_mean, image_std, tokenizer=None, shuffle=None):
    train = (split == "train")
    ds = FFADataset(C.H5[split], image_mean, image_std, train=train,
                    tokenizer=tokenizer if train else None)
    if shuffle is None:
        shuffle = train
    return torch.utils.data.DataLoader(
        ds, batch_size=C.BATCH_SIZE, shuffle=shuffle,
        num_workers=C.NUM_WORKERS, pin_memory=True, drop_last=train)
