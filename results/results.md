# Training Results

Run config: 5000 steps, 10.80M params (6 layers, 6 heads, 384 embedding dim,
256-token context), Apple Silicon MPS, ~3 hours wall clock (`train.py`
defaults, no changes).

## Loss curve

| step | train loss | val loss | wall time |
| ---- | ---------- | -------- | --------- |
| 0    | 4.2695     | 4.2675   | 68.6s     |
| 250  | 2.4273     | 2.4424   | 540.7s    |
| 500  | 2.0898     | 2.1600   | 1005.6s   |
| 750  | 1.7457     | 1.9061   | 1566.1s   |
| 1000 | 1.5587     | 1.7383   | 2218.9s   |
| 1250 | 1.4500     | 1.6500   | 2800.7s   |
| 1500 | 1.3681     | 1.5857   | 3305.8s   |
| 1750 | 1.3076     | 1.5399   | 3808.6s   |
| 2000 | 1.2587     | 1.5064   | 4294.1s   |
| 2250 | 1.2252     | 1.4880   | 4811.3s   |
| 2500 | 1.1909     | 1.4708   | 5340.1s   |
| 2750 | 1.1619     | 1.4642   | 5891.4s   |
| 3000 | 1.1342     | 1.4620   | 6463.7s   |
| 3250 | 1.1051     | 1.4530   | 7013.7s   |
| 3500 | 1.0835     | 1.4503   | 7543.9s   |
| 3750 | 1.0557     | 1.4665   | 8155.3s   |
| 4000 | 1.0251     | 1.4692   | 8580.1s   |
| 4250 | 0.9989     | 1.4690   | 9054.9s   |
| 4500 | 0.9735     | 1.4790   | 9568.3s   |
| 4750 | 0.9462     | 1.4953   | 10102.5s  |
| 4999 | 0.9188     | 1.4970   | 10655.1s  |

## Analysis

Val loss bottoms out at step 3500 (1.4503) and creeps back up for the rest
of training while train loss keeps falling — the model overfits past that
point. `train.py` saves a checkpoint at every eval step, so in practice the
step-3500 checkpoint (not the final one) is the one to keep for best
generalization. This wasn't tuned around; it's the honest behavior of the
default config with no early stopping or LR schedule.

## Sample output

`python sample.py`, prompt `"ROMEO:"`, temperature 0.8, top_k 40, checkpoint
from the final (step 4999) weights:

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

Correct play-script formatting (character names, dialogue structure, colons,
blank lines between speakers) emerges purely from character-level
next-token prediction — none of that structure was hardcoded anywhere in
the model or training loop.

## Earlier smoke-test run

Before the full run, a 300-step run on a much smaller config (block_size 64,
n_embd 64, n_head 4, n_layer 4, ~0.21M params) was used to sanity-check the
full pipeline end-to-end in under 15 seconds:

```
device: mps  params: 0.21M
step 0: train loss 4.2013, val loss 4.1992, time 1.5s
step 100: train loss 2.9596, val loss 2.9558, time 5.9s
step 200: train loss 2.6884, val loss 2.6985, time 8.5s
step 299: train loss 2.5688, val loss 2.5756, time 11.1s
```
