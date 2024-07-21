import math
import random
from enum import Enum
from typing import Callable, List, Tuple, Union

import selfies as sf
import torch
from rdkit import Chem
from torch import Tensor


class Randomize(Enum):
    """
    Enum representing different randomization options.

    Attributes:
        NONE (int): No randomization.
        RANDOM (int): Randomize the data.
        SRC_TARG (int): Create a source/target pair with randomization (encoder takes in source, decoder takes in target).
    """

    NONE = 1
    RANDOM = 2
    SRC_TARG = 3


def collate_selfies(batch: List[Tensor], pad_idx: int) -> Tensor:
    tokens = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_idx)

    return tokens


def encode_selfie(example: str, vocab: dict) -> Tensor:
    tokens = [f"[{t}" for t in example.split("[") if t != ""]
    tokens = ["[START]", *tokens, "[STOP]"]
    tokens = [vocab[t] for t in tokens]
    return torch.tensor(tokens)


def randomize_selfie(selfie: str, p: float = 1.0) -> str:
    if random.random() > p:
        return selfie

    smile = sf.decoder(selfie)
    mol = Chem.MolFromSmiles(smile)
    smile = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
    return sf.encoder(smile)


def randomize_batch(
    selfies: List[str], mode: Randomize, p: float = 1.0
) -> Union[List[str], Tuple[List[str], List[str]]]:
    if mode == Randomize.NONE:
        return selfies
    elif mode == Randomize.RANDOM:
        return [randomize_selfie(s, p=p) for s in selfies]
    elif mode == Randomize.SRC_TARG:
        src = [randomize_selfie(s, p=p) for s in selfies]
        targ = [randomize_selfie(s, p=p) for s in selfies]
        return src, targ
    else:
        raise ValueError(f"Unsupported randomize option: {mode}")


def build_transform(vocab: dict, randomize: Randomize) -> Callable:
    def transform(batch: List[dict]) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        try:
            selfies = [ex["SELFIE"] for ex in batch]
        except KeyError:
            selfies = [ex["SELFIES"] for ex in batch]

        if randomize == Randomize.NONE:
            tokens = [encode_selfie(s, vocab) for s in selfies]
            tokens = collate_selfies(tokens, vocab["[PAD]"])
            return tokens
        elif randomize == Randomize.RANDOM:
            selfies = [randomize_selfie(s, p=0.5) for s in selfies]
            tokens = [encode_selfie(s, vocab) for s in selfies]
            tokens = collate_selfies(tokens, vocab["[PAD]"])
            return tokens
        elif randomize == Randomize.SRC_TARG:
            src = [randomize_selfie(s, p=1.0 - 1 / math.sqrt(2)) for s in selfies]
            targ = [randomize_selfie(s, p=1.0 - 1 / math.sqrt(2)) for s in selfies]
            src_tokens = [encode_selfie(s, vocab) for s in src]
            targ_tokens = [encode_selfie(s, vocab) for s in targ]
            src_tokens = collate_selfies(src_tokens, vocab["[PAD]"])
            targ_tokens = collate_selfies(targ_tokens, vocab["[PAD]"])
            return src_tokens, targ_tokens
        else:
            raise ValueError(f"Invalid randomization mode: {randomize}")

    return transform
