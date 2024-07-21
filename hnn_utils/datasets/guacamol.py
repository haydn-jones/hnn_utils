import json

import datasets
import lightning as L
import pkg_resources
from torch.utils.data import DataLoader

from hnn_utils.datasets.utils import (
    Randomize,
    build_transform,
)

DS_PATH = "haydn-jones/Guacamol"


class GuacamolDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int, num_workers: int, randomize: Randomize = Randomize.NONE) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        vocab_path = "vocab/guac.json" if randomize == Randomize.NONE else "vocab/guac_random.json"
        self.vocab = json.load(pkg_resources.resource_stream(__name__, vocab_path))

        self.randomize = randomize

    def prepare_data(self) -> None:
        datasets.load_dataset(DS_PATH, split="train")
        datasets.load_dataset(DS_PATH, split="val")
        datasets.load_dataset(DS_PATH, split="test")

    def train_dataloader(self) -> DataLoader:
        ds: datasets.Dataset = datasets.load_dataset(DS_PATH, split="train").select_columns("SELFIE")  # type: ignore
        transform = build_transform(self.vocab, self.randomize)

        return DataLoader(
            ds,  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=transform,
        )

    def val_dataloader(self) -> DataLoader:
        ds: datasets.Dataset = datasets.load_dataset(DS_PATH, split="val").select_columns("SELFIE")  # type: ignore
        transform = build_transform(self.vocab, self.randomize)

        return DataLoader(
            ds,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=transform,
        )

    def test_dataloader(self) -> DataLoader:
        ds: datasets.Dataset = datasets.load_dataset(DS_PATH, split="test").select_columns("SELFIE")  # type: ignore
        transform = build_transform(self.vocab, self.randomize)

        return DataLoader(
            ds,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=transform,
        )
