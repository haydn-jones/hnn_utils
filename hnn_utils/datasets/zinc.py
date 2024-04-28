import json

import datasets
import lightning as L
import pkg_resources
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader

from hnn_utils.datasets.utils import (
    Randomize,
    build_transform,
)

DS_PATH = "haydn-jones/ZINC20"


class ZINC20DataModule(L.LightningDataModule):
    def __init__(
        self, batch_size: int, num_workers: int, randomize: Randomize = Randomize.NONE
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.vocab = json.load(
            pkg_resources.resource_stream(__name__, "vocab/zinc20.json")
        )

        self.randomize = randomize

    def prepare_data(self) -> None:
        datasets.load_dataset(DS_PATH, streaming=True, save_infos=True, split="train")
        datasets.load_dataset(DS_PATH, streaming=True, save_infos=True, split="val")
        datasets.load_dataset(DS_PATH, streaming=True, save_infos=True, split="test")

    def train_dataloader(self) -> DataLoader:
        ds = datasets.load_dataset(DS_PATH, streaming=True).select_columns("SELFIES")
        ds = split_dataset(ds, "train", self.trainer)

        transform = build_transform(self.vocab, self.randomize)

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=transform,
        )

    def val_dataloader(self) -> DataLoader:
        ds = datasets.load_dataset(DS_PATH, streaming=True).select_columns("SELFIES")
        ds = split_dataset(ds, "val", self.trainer)

        transform = build_transform(self.vocab, self.randomize)

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=transform,
        )

    def test_dataloader(self) -> DataLoader:
        ds = datasets.load_dataset(DS_PATH, streaming=True).select_columns("SELFIES")
        ds = split_dataset(ds, "test", self.trainer)

        transform = build_transform(self.vocab, self.randomize)

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=transform,
        )


def split_dataset(
    dataset: datasets.Dataset, split: str, trainer: L.Trainer
) -> datasets.Dataset:
    if trainer is None:
        return dataset

    dataset = dataset[split]
    return split_dataset_by_node(
        dataset, rank=trainer.global_rank, world_size=trainer.world_size
    )
