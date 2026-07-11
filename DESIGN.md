# Design Doc: mini-gpt

## Goal

Implement a decoder-only transformer language model from first principles —
no `transformers` library — well enough to explain every line, and train it
to a point where it produces recognizable output. The target is
understanding, not state-of-the-art results.

## Scope decisions

**Decoder-only, not encoder-decoder.** Language modeling (predict the next
token) only needs causal self-attention over a single sequence. An
encoder-decoder (like the original "Attention Is All You Need" translation
model) adds cross-attention and a second stack for a task this project isn't
doing. GPT-style decoder-only is the right amount of architecture for the
goal.

**Character-level tokenization, not BPE.** A byte-pair-encoding tokenizer
(what GPT-2/3 actually use) adds a whole subsystem — merge rules, a learned
vocabulary — that's orthogonal to the transformer architecture itself.
Character-level keeps the vocabulary trivial (65 symbols for this dataset)
so the model's behavior is easy to inspect, at the cost of longer sequences
per unit of text and a harder learning problem per step. This is the
standard tradeoff educational GPT implementations make.

**Tiny Shakespeare, not a large corpus.** ~1M characters is small enough to
memmap and iterate over quickly on a laptop, and Shakespeare's dialogue
structure (character names, colons, verse) gives a visually obvious signal
of whether the model has learned anything beyond "plausible English
letters."

**~10M parameters (6 layers, 6 heads, 384 dim, 256 context).** Sized to
train in a few hours on Apple Silicon MPS / a laptop CPU, not a GPU cluster.
Enough capacity to produce structured, mostly-grammatical text; nowhere near
enough to be coherent over long spans or to "know" facts.

## Architecture choices

- **Pre-norm residual blocks** (`x = x + Attn(LN(x))`) rather than post-norm.
  Pre-norm is what modern GPT implementations use because it trains more
  stably at this depth without needing a learning-rate warmup schedule as
  finely tuned as post-norm requires.
- **Learned absolute positional embeddings**, not rotary (RoPE) or ALiBi.
  Simpler to implement and reason about; the tradeoff is the model can't
  generalize to sequence lengths longer than `block_size` at inference time,
  which doesn't matter here since generation always happens within that
  window.
- **Single fused QKV projection** (`nn.Linear(n_embd, 3 * n_embd)`) rather
  than three separate projections — mathematically identical, fewer
  matmuls.
- **No weight tying** between the token embedding and the output head.
  GPT-2 ties these to save parameters; skipped here for clarity, since
  parameter count isn't the constraint at this scale.

## What was cut, and why

- **No BPE tokenizer** — see above.
- **No KV-cache for generation** — `generate()` recomputes attention over
  the full context window on every new token. This is O(n^2) instead of
  O(n) per generated token, which matters at production scale and doesn't
  matter for generating a few hundred characters in a demo.
- **No mixed precision / gradient accumulation / distributed training** —
  none of these are relevant at 10M params on a single machine; they'd add
  code without teaching anything the core architecture doesn't already
  cover.
- **No learning-rate schedule or early stopping** — deliberately left out so
  the [results](results/results.md) show the model's *actual* behavior
  (including the mild overfitting after step 3500) rather than a tuned
  result that hides it.

## Known limitations / future work

- Val loss overfits past step ~3500 — see [results/results.md](results/results.md).
  Adding a cosine LR decay and/or early stopping at the val-loss minimum
  would fix this cheaply.
- Character-level tokenization is inefficient; a BPE tokenizer would let
  the same context window cover much more text.
- No KV-cache means generation is slower than it needs to be for longer
  outputs.
- Single dataset (Tiny Shakespeare) — no evaluation on out-of-domain text.
