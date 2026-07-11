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

## Results

Full training run (5000 steps, 10.80M params, on Apple Silicon MPS, ~3 hours):

| step | train loss | val loss |
| ---- | ---------- | -------- |
| 0    | 4.2695     | 4.2675   |
| 500  | 2.0898     | 2.1600   |
| 1000 | 1.5587     | 1.7383   |
| 1500 | 1.3681     | 1.5857   |
| 2000 | 1.2587     | 1.5064   |
| 2500 | 1.1909     | 1.4708   |
| 3000 | 1.1342     | 1.4620   |
| 3500 | 1.0835     | 1.4503   |
| 4000 | 1.0251     | 1.4692   |
| 4500 | 0.9735     | 1.4790   |
| 4999 | 0.9188     | 1.4970   |

Val loss bottoms out around step 3500 (1.4503) and creeps back up while train
loss keeps falling — the model starts mildly overfitting past that point.
A checkpoint saved around step 3000-3500 generalizes best.

Sample output (`python sample.py`, prompt `"ROMEO:"`, temperature 0.8):

```text
ROMEO:
Nay, my lord, what says there?

POMPEY:
Truly, sir, good father.

MENENIUS:
I do beseech you, sir. Hear me to my wife:
Here comes the winds of the clouds of Bolingbroke,
And seeing it strength in secret post:
I'll try him for the good city and thee.

CORIOLANUS:
I shall not be too much of mine hair, not thy nose
Than my man's mouth: my general tomb!

Volsce:
What, shall I say?

Roman:
There is a man I was a tender than a mind
of the state?

VOLUMNIA:
Why, if it be not so, you have a wife, I say
```

Correct play-script formatting (character names, dialogue structure) emerges
purely from character-level next-token prediction — no rules were hardcoded.

## Why build this

Understanding a transformer means being able to write the attention math,
the causal mask, and the training loop yourself — not just importing
`AutoModel.from_pretrained(...)`. This repo is that exercise.
