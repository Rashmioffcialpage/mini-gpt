# mini-gpt

A minimal decoder-only transformer (GPT-style language model) implemented from
scratch in PyTorch — no `transformers` library, no shortcuts. Trained
character-by-character on the Tiny Shakespeare dataset.

## Architecture

`model.py` implements the standard GPT decoder stack:

- **Token + positional embeddings** — learned embeddings for each character
  and each position in the context window.
- **Causal self-attention** (`CausalSelfAttention`) — scaled dot-product
  multi-head attention with a lower-triangular mask, so position `t` can only
  attend to positions `<= t`.
- **MLP** — a 4x-widening feed-forward block with GELU activation.
- **Block** — pre-norm residual wrapping of attention + MLP
  (`x = x + Attn(LN(x))`, then `x = x + MLP(LN(x))`).
- **GPT** — stacks `n_layer` blocks, followed by a final LayerNorm and a
  linear head projecting back to vocabulary logits.

Default config: 6 layers, 6 heads, 384 embedding dim, 256-token context
(~10M parameters) — small enough to train on a laptop CPU/MPS in a
reasonable time, large enough to produce recognizably Shakespeare-like text.

## Usage

```bash
pip install -r requirements.txt

# tokenize the dataset and build train/val splits
python data/prepare.py

# train (checkpoints written to out/ckpt.pt)
python train.py

# generate text from the trained checkpoint
python sample.py
```

## Why build this

Understanding a transformer means being able to write the attention math,
the causal mask, and the training loop yourself — not just importing
`AutoModel.from_pretrained(...)`. This repo is that exercise.
