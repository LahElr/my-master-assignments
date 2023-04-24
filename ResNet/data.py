import pickle
import random
from typing import Any, Tuple
import torch
import torchvision
import torchvision.transforms as transforms
import utils
import os
import gc
import numpy

# these are commonly used data augmentations
# random cropping and random horizontal flip
# lastly, we normalize each channel into zero mean and unit standard deviation
train_transform_compose = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip()
]

train_transform_compose.append(transforms.ToTensor())

'''
p: probability that the random erasing operation will be performed.
scale: range of proportion of erased area against input image.
ratio: range of aspect ratio of erased area.
value: erasing value. Default is 0. If a single int, it is used to
   erase all pixels. If a tuple of length 3, it is used to erase
   R, G, B channels respectively.
   If a str of 'random', erasing each pixel with random values.
inplace: boolean to make this transform inplace. Default set to False.
'''
#! Here, I put the  RandomErasing` before `Normalize` because the (abs value of the) mean value of the Normalized figs are very small
#! Here, I put the `RandomErasing` after `ToTensor` because `RandomErasing` can only handle Tensors
#! The value has been changed after finding the problem to the scaled mean value
if utils.read_config("data.augment.use_erasing"):
    train_transform_compose.append(
        transforms.RandomErasing(p=utils.read_config("data.augment.erasing.p"), 
            scale=utils.read_config("data.augment.erasing.scale"), 
            ratio=utils.read_config("data.augment.erasing.ratio"), 
            value=utils.read_config("data.augment.erasing.value"))
    )
train_transform_compose.append(transforms.Normalize(
    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
transform_train = transforms.Compose(train_transform_compose)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_and_val_dataset = torchvision.datasets.CIFAR10(
    root=utils.read_config("data.path"), train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(
    root=utils.read_config("data.path"), train=False, download=True)

if not os.path.exists(os.path.join(utils.read_config("data.path"), "re_index")):
    os.makedirs(os.path.join(utils.read_config("data.path"), "re_index"))
    shuffled_index = list(range(len(train_and_val_dataset)))
    random.shuffle(shuffled_index)
    split_point = int(len(train_and_val_dataset) *
                      utils.read_config("data.train_val_ratio"))
    train_index = sorted(shuffled_index[:split_point])
    val_index = sorted(shuffled_index[split_point:])
    with open(os.path.join(utils.read_config("data.path"), "re_index", "train.pkl"), "wb") as pkl_file:
        pickle.dump(train_index, pkl_file)
    with open(os.path.join(utils.read_config("data.path"), "re_index", "val.pkl"), "wb") as pkl_file:
        pickle.dump(val_index, pkl_file)


class fake_list():
    def __init__(self, len):
        self.len = len

    def __getitem__(self, id):
        return id

    def __len__(self):
        return self.len


class NewCifer10(torch.utils.data.Dataset):
    def __init__(self, split: str = "train", transform: Any = None, target_transform: Any = None):
        if split == "train":
            self.split = 0
        elif split == "val" or split == "validate":
            split = "val"
            self.split = 1
        elif split == "test":
            self.split = 2
        else:
            raise ValueError(
                f"data.NewCifer10.__init__: unknown data split {split}")

        if self.split in (0, 1):
            self.data = train_and_val_dataset
        else:
            self.data = test_dataset

        if self.split != 2:
            with open(os.path.join(utils.read_config("data.path"), "re_index", f"{split}.pkl"), "rb") as pkl_file:
                self.index = pickle.load(pkl_file)
        else:
            self.index = fake_list(len(test_dataset))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, id: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[self.index[id]]

        # # doing this so that it is consistent with all other datasets
        # # to return a PIL Image
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.index)


gc.collect()
