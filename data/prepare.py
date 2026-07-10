"""Builds a character-level train/val split + vocab from data/input.txt."""

import os
import pickle

import numpy as np

DATA_DIR = os.path.dirname(__file__)
INPUT_PATH = os.path.join(DATA_DIR, "input.txt")


def main():
    with open(INPUT_PATH, "r") as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encoded = np.array([stoi[c] for c in text], dtype=np.uint16)

    split = int(0.9 * len(encoded))
    train_ids = encoded[:split]
    val_ids = encoded[split:]

    train_ids.tofile(os.path.join(DATA_DIR, "train.bin"))
    val_ids.tofile(os.path.join(DATA_DIR, "val.bin"))

    with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "stoi": stoi, "itos": itos}, f)

    print(f"vocab_size: {vocab_size}")
    print(f"train tokens: {len(train_ids):,}  val tokens: {len(val_ids):,}")


if __name__ == "__main__":
    main()
