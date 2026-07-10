import os
import pickle
import time

import numpy as np
import torch

from model import GPT

# ---- config ----
data_dir = os.path.join(os.path.dirname(__file__), "data")
out_dir = "out"
block_size = 256
batch_size = 64
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
learning_rate = 3e-4
max_iters = 5000
eval_interval = 250
eval_iters = 100
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

torch.manual_seed(1337)

# ---- data ----
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
vocab_size = meta["vocab_size"]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
    )
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    os.makedirs(out_dir, exist_ok=True)

    model = GPT(vocab_size, block_size, n_embd, n_head, n_layer, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"device: {device}  params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    t0 = time.time()
    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model)
            print(
                f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
                f"time {time.time() - t0:.1f}s"
            )
            torch.save(model.state_dict(), os.path.join(out_dir, "ckpt.pt"))

        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()
