# import pickle
# import random
from typing import List, Tuple
import torch
import utils
import os
import torchvision.transforms as transforms
from PIL import Image
import transformers


def read_split(split: str):
    file_path = utils.read_config("data.path")
    imgs = os.path.join(file_path, "split", f"{split}.txt")
    with open(imgs) as img_file:
        imgs = [os.path.split(_.strip())[-1] for _ in img_file.readlines()]
    if split == "test":
        attrs = None
    else:
        attrs = os.path.join(file_path, "split", f"{split}_attr.txt")
        with open(attrs) as attrs_file:
            attrs = [list(map(int, _.split())) for _ in attrs_file.readlines()]
        assert len(imgs) == len(attrs)
    return imgs, attrs


class PreprocessCollater():
    def __init__(self):
        self.image_process = transformers.ConvNextImageProcessor(
            **utils.read_config("data.image_process"))

    def __call__(self, batch):
        '''
        eg:
        [
            (
                <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=200x300 at 0x1909C76BD08>, 
                [5, 0, 2, 0, 2, 2]
            ), 
            (
                <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=300x251 at 0x1909C772848>, 
                [5, 1, 2, 0, 5, 1]
            )
        ]
        '''
        imgs = [_[0] for _ in batch]
        attrs = [_[1] for _ in batch]
        imgs = self.image_process(imgs, return_tensors="pt")[
            'pixel_values']  #{"pixel_values":tensor of size [1,3,224,224]}
        if attrs[0] is not None:
            attrs = torch.tensor(attrs)
        else:
            attrs = None
        return [imgs, attrs]


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, split: str = "val"):
        if split not in ["test", "val"]:
            raise ValueError(
                f"data.TestDataset.__init__: This is the dataset for validation and testing, but got {split}."
            )
        self.imgs, self.attrs = read_split(split)
        self.length = len(self.imgs)

    def __getitem__(self, id: int) -> Tuple[torch.Tensor, List[int]]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img = self.imgs[id]
        img = Image.open(
            os.path.join(utils.read_config("data.path"), "img",
                         f"{img}"))  # PIL.JpegImagePlugin.JpegImageFile
        if self.attrs is not None:
            attr = self.attrs[id]
        else:
            attr = None
        return img, attr

    def __len__(self):
        return self.length


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.imgs, self.attrs = read_split("train")
        self.length = len(self.attrs)
        train_transform_compose = []

        if utils.read_config("data.augment.use_random_crop"):
            train_transform_compose.append(
                transforms.RandomCrop(
                    **utils.read_config("data.augment.random_crop_args")))

        train_transform_compose.append(
            transforms.RandomHorizontalFlip(
                p=utils.read_config("data.augment.random_horizontal_flip")))

        train_transform_compose.append(transforms.ToTensor())
        #! Here, I put the  RandomErasing` before `Normalize` because the (abs value of the) mean value of the Normalized figs are very small
        #! Here, I put the `RandomErasing` after `ToTensor` because `RandomErasing` can only handle Tensors
        #! The value has been changed after finding the problem to the scaled mean value
        if utils.read_config("data.augment.use_erasing"):
            train_transform_compose.append(
                transforms.RandomErasing(
                    p=utils.read_config("data.augment.erasing.p"),
                    scale=utils.read_config("data.augment.erasing.scale"),
                    ratio=utils.read_config("data.augment.erasing.ratio"),
                    value=utils.read_config("data.image_process.image_mean")))
            train_transform_compose.append(transforms.ToPILImage())
        # train_transform_compose.append(
        #     transforms.Normalize(
        #         utils.read_config("data.image_process.image_mean"),
        #         utils.read_config("data.image_process.image_std")))
        self.transform = transforms.Compose(train_transform_compose)

    def __getitem__(self, id: int) -> Tuple[torch.Tensor, List[int]]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img = self.imgs[id]
        img = Image.open(
            os.path.join(utils.read_config("data.path"), "img",
                         f"{img}"))  # PIL.JpegImagePlugin.JpegImageFile
        img = self.transform(img)  # out is still an pil image

        # attr = torch.tensor(self.attrs[id])
        attr = self.attrs[id]
        return img, attr

    def __len__(self):
        return self.length


def get_traindata():
    '''
    Each batch: [torch.Size([bs, 3, 224, 224]),torch.Size([32, 6])]
    '''
    dataset = TrainDataset()
    collate_function = PreprocessCollater()
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=utils.read_config("train.batch_size"),
        shuffle=True,
        num_workers=utils.read_config("train.num_workers"),
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_function)
    return dataset, dataloader


def get_testdata(split):
    '''
    Each batch: [torch.Size([bs, 3, 224, 224]),torch.Size([32, 6])]
    '''
    dataset = TestDataset(split)
    collate_function = PreprocessCollater()
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=utils.read_config("train.batch_size"),
        shuffle=False,
        num_workers=utils.read_config("train.num_workers"),
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_function)
    return dataset, dataloader


# gc.collect()
