import json
from typing import Optional

import datasets
import lightning as L
import pkg_resources
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader

from hnn_utils.datasets.utils import (
    build_transform,
    Randomize,
)


DS_PATH = "haydn-jones/ZINC20"


class ZINC20DataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        randomize: Randomize = Randomize.NONE,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.vocab = json.load(pkg_resources.resource_stream(__name__, "vocab/zinc20.json"))

        self.randomize = randomize
        self.seed = seed

        self.dataset: Optional[datasets.DatasetDict] = None

    def prepare_data(self) -> None:
        datasets.load_dataset(DS_PATH, streaming=True, save_infos=True)

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = datasets.load_dataset(DS_PATH, streaming=True)  # type: ignore

    def train_dataloader(self) -> DataLoader:
        ds = self.dataset["train"].select_columns("SELFIES")  # type: ignore
        ds = split_and_shuffle(ds, self.trainer, seed=self.seed)  # type: ignore

        transform = build_transform(self.vocab, self.randomize)

        return DataLoader(
            ds,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=transform,
        )

    def val_dataloader(self) -> DataLoader:
        ds = self.dataset["val"].select_columns("SELFIES")  # type: ignore
        ds = split_and_shuffle(ds, self.trainer, seed=self.seed)  # type: ignore

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
        ds = self.dataset["test"].select_columns("SELFIES")  # type: ignore
        ds = split_and_shuffle(ds, self.trainer, seed=self.seed)  # type: ignore

        transform = build_transform(self.vocab, self.randomize)

        return DataLoader(
            ds,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=transform,
        )


def split_and_shuffle(
    dataset: datasets.IterableDataset,
    trainer: Optional[L.Trainer],
    seed: int = 42,
) -> datasets.IterableDataset:
    # Split according to distributed setup
    if trainer is not None:
        dataset = split_dataset_by_node(dataset, rank=trainer.global_rank, world_size=trainer.world_size)

    # Shuffling an iterable dataset shuffles the *shards* and
    # creates a buffer of size `buffer_size` (in elements, not bytes)
    # which it randomly samples from. Larger buffer_size
    # means more memory usage but better shuffling.
    dataset = dataset.shuffle(buffer_size=16384, seed=seed)

    return dataset
