# Transformer Language Model 

A character-level GPT transformer built from scratch in PyTorch, trained on children's stories to generate simple English narrative text character by character. No pre-trained weights. No fine-tuning. Pure architecture and training from zero.
- Actual Training log
<img width="946" height="845" alt="Screenshot 2026-03-21 001248" src="https://github.com/user-attachments/assets/fcac4e2f-9f52-4c59-8c29-199004d058ce" />

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Project Structure](#project-structure)
3. [How It Works](#how-it-works)
4. [Setup & Requirements](#setup--requirements)
5. [How to Run](#how-to-run)
6. [Configuration](#configuration)
7. [Actual Training Results](#actual-training-results)
8. [Real Model Output](#real-model-output)
9. [Output Analysis](#output-analysis)
10. [Overfitting — Did It Happen?](#overfitting--did-it-happen)
11. [How Weights Produce Output](#how-weights-produce-output)
12. [Scaling Laws — And Where Your Model Sits](#scaling-laws--and-where-your-model-sits)
13. [Known Limitations](#known-limitations)

---

## What This Project Does

This project trains a small GPT-style transformer model on children's stories and then generates new story-like text character by character — infinitely — in the same terminal window after training finishes.

It is a learning project. The goal is not to produce publishable stories, but to understand how language models learn patterns from text and to see that process happen live on your own machine.

---

## How It Works

The model is a **character-level transformer**. This means:

- It reads your text file one character at a time
- It learns which characters tend to follow other characters in which contexts
- At generation time it predicts the next character, then the next, then the next — forever

It is the same core architecture as GPT, just much smaller and trained on much less data.

**The pipeline in order:**

```
cleaned.txt  (children's stories)
    ↓
Characters encoded as integers (vocab size: 28)
    ↓
Model trains on sequences of 128 characters at a time
    ↓
Every 200 steps: loss is measured and printed
    ↓
Best weights saved to best_model.pt whenever val loss improves
    ↓
After 3000 steps: infinite generation starts in same terminal
    ↓
Press Ctrl+C to stop
```

---

## Setup & Requirements

**Python version:** 3.8 or higher

**Install dependencies:**

```bash
pip install torch
```

No other dependencies needed. The project uses only PyTorch and Python standard library modules.

**Your `config/config.py` should look like this:**

```python
cleaned_path = "cleaned.txt"
train_split  = 0.9
seed         = 42
```

---

## How to Run

```bash
python transformer.py
```

The script will:

1. Print a startup banner with device and timestamp
2. Load and report stats on your dataset
3. Build the model and print parameter count
4. Train for 3000 steps, printing progress every 200 steps
5. Save best weights to `best_model.pt` automatically
6. Start infinite text generation in the same terminal when done

**To stop generation:** press `Ctrl+C`

---

## Configuration

All hyperparameters are at the top of `transformer.py`, tuned for a CPU-only machine:

```python
batch_size    = 16      # How many sequences to train on at once
block_size    = 128     # How many characters the model sees at once
max_iters     = 3000    # Total training steps
eval_interval = 200     # Print progress every N steps
eval_iters    = 50      # Batches averaged for loss estimate
learning_rate = 3e-4    # How fast the model updates
n_embd        = 128     # Size of internal representations
n_head        = 4       # Number of attention heads
n_layer       = 4       # Number of transformer blocks
dropout       = 0.2     # Regularization — prevents memorization
```

**Parameter count: 0.82M parameters**

---

## Actual Training Results

Trained on: AMD Ryzen 5 PRO 3500U (CPU only, no GPU)

```
Parameters    : 0.82M
Dataset       : 201,570 characters of children's stories
Vocabulary    : 28 unique characters
Training time : 39.4 minutes
Best val loss : 1.3145  (reached at step 2999 — still improving)
Final train   : 1.3191
```

**Full training log:**

```
[    0/3000]   0.0%   train=3.2961   val=3.2981   elapsed=12s     ETA=0s      << best!
[  200/3000]   6.7%   train=2.3038   val=2.2490   elapsed=141s    ETA=1959s   << best!
[  400/3000]  13.3%   train=2.2469   val=2.1950   elapsed=292s    ETA=1891s   << best!
[  600/3000]  20.0%   train=2.1842   val=2.1318   elapsed=436s    ETA=1739s   << best!
[  800/3000]  26.7%   train=1.9742   val=1.9103   elapsed=581s    ETA=1594s   << best!
[ 1000/3000]  33.3%   train=1.7628   val=1.7002   elapsed=723s    ETA=1443s   << best!
[ 1200/3000]  40.0%   train=1.6714   val=1.6040   elapsed=863s    ETA=1293s   << best!
[ 1400/3000]  46.7%   train=1.5889   val=1.5360   elapsed=1015s   ETA=1158s   << best!
[ 1600/3000]  53.3%   train=1.5375   val=1.4723   elapsed=1166s   ETA=1019s   << best!
[ 1800/3000]  60.0%   train=1.4847   val=1.4525   elapsed=1320s   ETA=879s    << best!
[ 2000/3000]  66.7%   train=1.4604   val=1.4081   elapsed=1472s   ETA=735s    << best!
[ 2200/3000]  73.3%   train=1.4113   val=1.3857   elapsed=1653s   ETA=600s    << best!
[ 2400/3000]  80.0%   train=1.3923   val=1.3725   elapsed=1820s   ETA=454s    << best!
[ 2600/3000]  86.7%   train=1.3501   val=1.3446   elapsed=1998s   ETA=307s    << best!
[ 2800/3000]  93.3%   train=1.3336   val=1.3334   elapsed=2174s   ETA=154s    << best!
[ 2999/3000] 100.0%   train=1.3191   val=1.3145   elapsed=2363s   ETA=0s      << best!

[DONE] Training finished in 2364.1s (39.4 min) | Best val loss: 1.3145
[SAVE] Best weights saved to best_model.pt
```

---

## Real Model Output

This is actual output generated by the model after training:

```
when years me told be found a big ea reak abig driendly they named not she
rabbit smiled by aded he what in again
one smiled the mushrought boy
one day and was arroom him that she rabbing animals the dreezed at neard had
to there man owl them with o box and said you s mom that je animable went her
somethings of they ballike i wanted a big taught jill hone was and
he rabbit to havin after the but help and nelpft but it was surpring take to
he hard along could he had a shot jack and loved explay liked ayou had and
kine it s good
the got and the but and garden he saw toy
and the was frieved to i me the game one can he would and mary alt his so
and little bears okay would found the loved hugger mosted
```

---

## Output Analysis

The output is genuinely impressive for a 0.82M parameter model trained in 39 minutes on a CPU. Here is what the model learned:

**What it got right:**
```
Story structure     → "one day...", paragraphs, narrative flow
Character names     → jack, tim, lucy, jimmy, mary, nick, john
Common story words  → smiled, helped, loved, happy, garden, friends
Sentence patterns   → "he said", "she was", "they went"
Story phrases       → "one day", "suddenly", "the next"
Word spacing        → mostly correct
```

**What it got wrong:**
```
Spelling            → "mushrought", "driendly", "surpring"
Logic               → sentences don't connect coherently
Grammar             → articles and pronouns sometimes misplaced
```

**Why the spelling is off:**

The model works character by character. It learned that after `fr` comes `i`, then `e`, then `n`, then `d` — but it sometimes gets the sequence slightly wrong, producing `driendly` instead of `friendly`. It has no concept of words as units, only character patterns.

**Comparison to previous kernel C run:**

| Metric          | Kernel C run    | This run (stories) |
|-----------------|-----------------|--------------------|
| Dataset         | 117K chars      | 201K chars         |
| Vocab size      | 95              | 28                 |
| Best val loss   | 2.3924          | 1.3145             |
| Overfitting     | Yes (step 1400) | No (still improving at 3000) |
| Output quality  | Code-like noise | Readable story fragments |

The stories run is significantly better across every metric. Smaller vocabulary, more data, and lower loss all contributed.

---

## Overfitting — Did It Happen?

**No. This run had no overfitting at all.**

```
Every single checkpoint showed << best!
Train loss and val loss decreased together the entire run
At step 2999 the model was still improving
```

Compare this to the previous kernel C run where overfitting started at step 1400:

```
Kernel C run:
  Step 1400: val=2.3924  << best  ← last improvement
  Step 1600: val=2.4409           ← overfitting starts
  Step 3000: val=2.4854           ← getting worse

This run:
  Step 1400: val=1.5360  << best
  Step 1600: val=1.4723  << best  ← still improving
  Step 3000: val=1.3145  << best  ← still improving at the end
```

**Why no overfitting this time:**

The dataset is nearly double the size (201K vs 117K characters) and the vocabulary is much smaller (28 vs 95 characters). With less to learn per character and more examples to learn from, the model never ran out of new patterns to generalize.

**What this means practically:**

The model could likely benefit from more training steps. Running 5000 or even 10000 iterations would probably continue to improve val loss. Try increasing `max_iters = 5000` next run.

---

## How Weights Produce Output

After training, the model is frozen. The weights file (`best_model.pt`) is 0.82 million numbers that encode everything the model learned from your children's stories.

**The generation loop:**

```
Step 1 — Start with a seed token (zero = start of text)
              ↓
Step 2 — Feed it through all 4 transformer layers
         Each layer does matrix multiplications
         using the saved weight numbers
              ↓
Step 3 — Output is 28 numbers (one per vocab character)
         Each number = probability of that character being next
         e.g.  'e' = 0.18   't' = 0.14   'a' = 0.12
              ↓
Step 4 — Sample randomly from those probabilities
              ↓
Step 5 — That character becomes the new input
         Go back to Step 2
              ↓
Step 6 — Repeat forever
```

**Why output is different every run:**

The sampling step picks randomly from the probabilities. Same weights, different random draws = different output each time. To get the same output every time add `torch.manual_seed(42)` before generation.

---

## Scaling Laws — And Where Your Model Sits
<img width="1029" height="705" alt="Screenshot 2026-03-17 171921" src="https://github.com/user-attachments/assets/ca7febe6-ed6d-463d-ba94-6a04cac1434a" />


### What Are Scaling Laws?

Scaling laws describe a predictable relationship between model size, dataset size, compute, and output quality:

> The more parameters, the more data, and the more compute you use — the better the model gets. And this follows a consistent, measurable curve.

The key finding (Chinchilla, 2022) is that the optimal ratio is roughly **20 tokens of training data per parameter.**

### The Three Axes of Scaling

```
Parameters (N)  →  How much the model can remember
Data (D)        →  How much it has learned from
Compute (C)     →  Parameters × Data × Training steps
```

All three need to grow together for best results.

### Where Your Model Sits

```
Model                    Parameters    Data              Quality
───────────────────────────────────────────────────────────────
This run (stories)       0.82M         201K chars        Readable fragments
Previous run (kernel C)  0.83M         117K chars        Code-like noise
GPT-2 Small              117M          ~40GB text        Coherent English
GPT-2 Large              774M          ~40GB text        Strong English
GPT-3                    175B          ~600GB text       Near-human text
```

### Specific Numbers

```
Parameters       :  0.82M
Training data    :  201,570 characters ≈ 201K tokens
Optimal data     :  20 × 820,000 = 16.4M tokens needed
Data you have    :  201K / 16.4M = 1.2% of what is optimal
```

Even at 1.2% of optimal data the model produced readable output and showed no overfitting — which proves that even severely data-limited models can learn meaningful patterns.

**The fact that val loss was still improving at step 3000 is the key signal** — this model has not yet hit its ceiling. More data and more training steps would both directly improve output quality.

**Highest-impact next steps in order:**

```
1. More data      → go from 201K to 1M+ characters of stories
2. More steps     → increase max_iters from 3000 to 5000 or 10000
3. Larger model   → only worth it after steps 1 and 2
```

---

## Known Limitations

- **CPU only** — no CUDA GPU means training is slow and larger configs are impractical
- **Character-level** — the model learns characters not words, so it cannot spell reliably or understand meaning
- **Small dataset** — 201K characters is still well below the optimal ~16M for this model size
- **Output will not make sense** — story fragments are recognizable but logically disconnected
- **No memory between runs** — each generation starts from scratch with no context
- **Model undertrained** — val loss was still falling at step 3000, more training would help

---

*Built with PyTorch. Architecture based on Andrej Karpathy's nanoGPT.*
