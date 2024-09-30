from pathlib import Path
from typing import Union
from random import shuffle

import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data import Subset

class DogBreedImageDataModule(L.LightningDataModule):
    def __init__(self, dl_path: Union[str, Path] = "data", num_workers: int = 0, batch_size: int = 8):
        super().__init__()
        self._dl_path = dl_path
        self._num_workers = num_workers
        self._batch_size = batch_size

    def prepare_data(self):
        """Download images and prepare images datasets."""
        dataset_path = self.data_path

        if not dataset_path.exists():
            download_and_extract_archive(
                url="https://github.com/m-shilpa/lightning-template-hydra/raw/main/dog_breed_image_dataset.zip",
                download_root=self._dl_path,
                remove_finished=True
            )

    @property
    def data_path(self):
        return Path(self._dl_path).joinpath("dataset")

    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    @property
    def valid_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize_transform
        ])

    def create_dataset(self, root, transform):
        return ImageFolder(root=root, transform=transform)

    def __dataloader(self, train: bool):
        """Train/validation/test loaders."""
        split_size =0.8
        full_dataset = self.create_dataset(self.data_path, self.train_transform)

        # indexes = list(range(len(full_dataset)))
        # shuffle(indexes)
        # indexes_train = indexes[:int(len(full_dataset)*split_size)]
        # indexes_test = indexes[int(len(full_dataset)*split_size):]

        indexes_train = range(int(len(full_dataset)*split_size))
        indexes_test = range(int(len(full_dataset)*split_size), len(full_dataset))

 
        if train:
            dataset = Subset(full_dataset, indexes_train)
        else:
            dataset = Subset(full_dataset, indexes_test)
        return DataLoader(dataset=dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=train)

    def train_dataloader(self):
        return self.__dataloader(train=True)

    def val_dataloader(self):
        return self.__dataloader(train=False)

    def test_dataloader(self):
        return self.__dataloader(train=False)  # Using validation dataset for testing