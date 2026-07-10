import os
import pickle

import torch

from model import GPT
from train import block_size, dropout, n_embd, n_head, n_layer

data_dir = os.path.join(os.path.dirname(__file__), "data")
ckpt_path = os.path.join("out", "ckpt.pt")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    stoi, itos, vocab_size = meta["stoi"], meta["itos"], meta["vocab_size"]

    model = GPT(vocab_size, block_size, n_embd, n_head, n_layer, dropout).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    prompt = "ROMEO:"
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)

    out = model.generate(idx, max_new_tokens=500, temperature=0.8, top_k=40)
    print("".join(itos[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
