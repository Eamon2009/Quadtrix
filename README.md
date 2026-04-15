# Quadtrix  ![GitHub](https://img.shields.io/github/license/Eamon2009/Transformer-language-model)
<img width="1138" height="515" alt="image" src="https://github.com/user-attachments/assets/cb665099-bdc0-4966-b857-f80a56a6642d" />
A character-level GPT transformer built from scratch in PyTorch, trained on children's stories to generate simple English narrative text character by character. No pre-trained weights. No fine-tuning. Pure architecture and training from zero.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Project Structure](#project-structure)
3. [How It Works](#how-it-works)
4. [Setup & Requirements](#setup--requirements)
5. [How to Run](#how-to-run)
6. [Configuration](#configuration)
7. [Training Runs — All Results](#training-runs--all-results)
8. [Head-to-Head Comparison](#head-to-head-comparison)
9. [Model Output Comparison](#model-output-comparison)
10. [Loss Curve Analysis](#loss-curve-analysis)
11. [Overfitting Analysis](#overfitting-analysis)
12. [Scaling Laws — And Where Your Model Sits](#scaling-laws--and-where-your-model-sits)
13. [How Weights Produce Output](#how-weights-produce-output)
14. [Known Limitations](#known-limitations)

---

## What This Project Does

This project trains a small GPT-style transformer model on children's stories and then generates new story-like text character by character. It is a learning project — the goal is not to produce publishable stories, but to understand how language models learn patterns from text and to see that process happen live on your own machine and cloud GPU.

---

## How It Works

The model is a **character-level transformer**. This means:

- It reads your text file one character at a time
- It learns which characters tend to follow other characters in which contexts
- At generation time it predicts the next character, then the next, then the next — forever

It is the same core architecture as GPT, just much smaller and trained on much less data.

**The pipeline in order:**

```
data.txt  (children's stories)
    ↓
Characters encoded as integers (vocab size varies by run)
    ↓
Model trains on sequences of tokens at a time
    ↓
Every N steps: loss is measured and printed
    ↓
Best weights saved to best_model.pt whenever val loss improves
    ↓
After training: text generation begins
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

---

## How to Run

```bash
python transformer.py
```

The script will:

1. Print a startup banner with device and timestamp
2. Load and report stats on your dataset
3. Build the model and print parameter count
4. Train for the configured number of steps, printing progress at each eval interval
5. Save best weights to `best_model.pt` automatically
6. Start text generation when done

---

## Configuration

### Run 3 — GPU Configuration (Tesla T4, Latest) ⭐

```python
batch_size    = 64      # Sequences trained on at once
block_size    = 128     # Context window (tokens)
max_iters     = 5000    # Total training steps
eval_interval = 200     # Print progress every N steps
learning_rate = 3e-4
n_embd        = 200     # Size of internal representations
n_head        = 4       # Number of attention heads
n_layer       = 4       # Number of transformer blocks
dropout       = 0.2
```

**Parameter count: 1.99M parameters**

### Run 2 — GPU Configuration (Google Colab)

```python
batch_size    = 64
block_size    = 256
max_iters     = 5000
eval_interval = 250
learning_rate = 3e-4
n_embd        = 384
n_head        = 6
n_layer       = 6
dropout       = 0.2
```

**Parameter count: 10.82M parameters**

### Run 1 — CPU Configuration (Laptop, Minimal Setup)

```python
batch_size    = 16
block_size    = 128
max_iters     = 3000
eval_interval = 200
learning_rate = 3e-4
n_embd        = 128
n_head        = 4
n_layer       = 4
dropout       = 0.2
```

**Parameter count: 0.82M parameters**

---

## Training Runs — All Results

### Run 3 — GPU (Tesla T4, 1.99M Parameters) ⭐ Latest

| Field | Value |
|---|---|
| Device | Tesla T4 (CUDA 13.0, Driver 580.82.07) |
| Dataset | ~31.4M characters |
| Vocab size | 100 |
| Train tokens | 28,274,093 |
| Val tokens | 3,141,566 |
| Parameters | 1.99M |
| Architecture | 4 layers × 4 heads × 200 embd dim |
| Training time | **6.1 minutes (367s)** |
| Best val loss | **0.9250** |
| Final train loss | 0.9307 |
| Overfitting | None — `best!` at most checkpoints |

**Full training log:**

```
[    0/5000]   train=4.6207   val=4.6202   elapsed=2s    ETA=0s    << best!
[  200/5000]   train=2.2058   val=2.1986   elapsed=17s   ETA=405s  << best!
[  400/5000]   train=1.6111   val=1.6039   elapsed=32s   ETA=367s  << best!
[  600/5000]   train=1.4109   val=1.4183   elapsed=47s   ETA=342s  << best!
[  800/5000]   train=1.3230   val=1.3231   elapsed=61s   ETA=322s  << best!
[ 1000/5000]   train=1.2495   val=1.2567   elapsed=76s   ETA=303s  << best!
[ 1200/5000]   train=1.1960   val=1.1948   elapsed=90s   ETA=286s  << best!
[ 1400/5000]   train=1.1569   val=1.1642   elapsed=105s  ETA=270s  << best!
[ 1600/5000]   train=1.1283   val=1.1283   elapsed=120s  ETA=254s  << best!
[ 1800/5000]   train=1.0894   val=1.1023   elapsed=134s  ETA=238s  << best!
[ 2000/5000]   train=1.0731   val=1.0765   elapsed=149s  ETA=223s  << best!
[ 2200/5000]   train=1.0584   val=1.0550   elapsed=163s  ETA=208s  << best!
[ 2400/5000]   train=1.0415   val=1.0346   elapsed=178s  ETA=192s  << best!
[ 2600/5000]   train=1.0261   val=1.0199   elapsed=192s  ETA=177s  << best!
[ 2800/5000]   train=1.0106   val=1.0117   elapsed=207s  ETA=162s  << best!
[ 3000/5000]   train=1.0000   val=0.9956   elapsed=221s  ETA=148s  << best!
[ 3200/5000]   train=0.9913   val=0.9924   elapsed=236s  ETA=133s  << best!
[ 3400/5000]   train=0.9727   val=0.9782   elapsed=251s  ETA=118s  << best!
[ 3600/5000]   train=0.9656   val=0.9720   elapsed=265s  ETA=103s  << best!
[ 3800/5000]   train=0.9685   val=0.9632   elapsed=280s  ETA=88s   << best!
[ 4000/5000]   train=0.9601   val=0.9642   elapsed=294s  ETA=74s
[ 4200/5000]   train=0.9515   val=0.9489   elapsed=309s  ETA=59s   << best!
[ 4400/5000]   train=0.9433   val=0.9431   elapsed=323s  ETA=44s   << best!
[ 4600/5000]   train=0.9384   val=0.9459   elapsed=338s  ETA=29s
[ 4800/5000]   train=0.9331   val=0.9250   elapsed=353s  ETA=15s   << best!
[ 4999/5000]   train=0.9307   val=0.9430   elapsed=367s  ETA=0s

[DONE] Training finished in 367.0s (6.1 min) | Best val loss: 0.9250
```

---

### Run 2 — GPU (Google Colab, 10.82M Parameters)

| Field | Value |
|---|---|
| Device | CUDA (Google Colab GPU) |
| Dataset | 88,406,739 characters |
| Vocab size | 110 |
| Train tokens | 79,566,065 |
| Val tokens | 8,840,674 |
| Parameters | 10.82M (10,823,534) |
| Architecture | 6 layers × 6 heads × 384 embd dim |
| Training time | **61.3 minutes** |
| Best val loss | **0.7176** |
| Final train loss | 0.7259 |
| Overfitting | None — `best!` at every checkpoint |

**Full training log:**

```
[    0/5000]   0.0%   train=4.9244   val=4.9262   elapsed=31s     ETA=0s      best!
[  250/5000]   5.0%   train=2.1218   val=2.1169   elapsed=206s    ETA=3901s   best!
[  500/5000]  10.0%   train=1.3606   val=1.3500   elapsed=391s    ETA=3510s   best!
[  750/5000]  15.0%   train=1.1540   val=1.1411   elapsed=575s    ETA=3250s   best!
[ 1000/5000]  20.0%   train=1.0332   val=1.0296   elapsed=757s    ETA=3024s   best!
[ 1250/5000]  25.0%   train=0.9657   val=0.9556   elapsed=941s    ETA=2819s   best!
[ 1500/5000]  30.0%   train=0.9305   val=0.9189   elapsed=1124s   ETA=2619s   best!
[ 1750/5000]  35.0%   train=0.8935   val=0.8853   elapsed=1306s   ETA=2424s   best!
[ 2000/5000]  40.0%   train=0.8673   val=0.8602   elapsed=1490s   ETA=2233s   best!
[ 2250/5000]  45.0%   train=0.8413   val=0.8367   elapsed=1672s   ETA=2042s   best!
[ 2500/5000]  50.0%   train=0.8162   val=0.8141   elapsed=1855s   ETA=1854s   best!
[ 2750/5000]  55.0%   train=0.8058   val=0.7995   elapsed=2038s   ETA=1666s   best!
[ 3000/5000]  60.0%   train=0.7888   val=0.7803   elapsed=2221s   ETA=1479s   best!
[ 3250/5000]  65.0%   train=0.7798   val=0.7730   elapsed=2403s   ETA=1293s   best!
[ 3500/5000]  70.0%   train=0.7634   val=0.7551   elapsed=2585s   ETA=1107s   best!
[ 3750/5000]  75.0%   train=0.7588   val=0.7528   elapsed=2768s   ETA=922s    best!
[ 4000/5000]  80.0%   train=0.7480   val=0.7434   elapsed=2951s   ETA=737s    best!
[ 4250/5000]  85.0%   train=0.7381   val=0.7351   elapsed=3134s   ETA=552s    best!
[ 4500/5000]  90.0%   train=0.7371   val=0.7314   elapsed=3316s   ETA=368s    best!
[ 4750/5000]  95.0%   train=0.7282   val=0.7239   elapsed=3498s   ETA=183s    best!
[ 4999/5000] 100.0%   train=0.7259   val=0.7176   elapsed=3680s   ETA=0s      best!

[DONE] Training finished in 3680.1s (61.3 min)
[DONE] Best val loss: 0.7176
[SAVE] Best weights saved to: /content/best_model.pt
```

---

### Run 1 — CPU (Laptop, 0.82M Parameters)

| Field | Value |
|---|---|
| Device | AMD Ryzen 5 PRO 3500U (CPU only) |
| Dataset | 201,570 characters |
| Vocab size | 28 |
| Parameters | 0.82M |
| Architecture | 4 layers × 4 heads × 128 embd dim |
| Training time | **39.4 minutes** |
| Best val loss | **1.3145** |
| Final train loss | 1.3191 |
| Overfitting | None — `best!` at every checkpoint |

**Full training log:**

```
[    0/3000]   0.0%   train=3.2961   val=3.2981   elapsed=12s     ETA=0s      best!
[  200/3000]   6.7%   train=2.3038   val=2.2490   elapsed=141s    ETA=1959s   best!
[  400/3000]  13.3%   train=2.2469   val=2.1950   elapsed=292s    ETA=1891s   best!
[  600/3000]  20.0%   train=2.1842   val=2.1318   elapsed=436s    ETA=1739s   best!
[  800/3000]  26.7%   train=1.9742   val=1.9103   elapsed=581s    ETA=1594s   best!
[ 1000/3000]  33.3%   train=1.7628   val=1.7002   elapsed=723s    ETA=1443s   best!
[ 1200/3000]  40.0%   train=1.6714   val=1.6040   elapsed=863s    ETA=1293s   best!
[ 1400/3000]  46.7%   train=1.5889   val=1.5360   elapsed=1015s   ETA=1158s   best!
[ 1600/3000]  53.3%   train=1.5375   val=1.4723   elapsed=1166s   ETA=1019s   best!
[ 1800/3000]  60.0%   train=1.4847   val=1.4525   elapsed=1320s   ETA=879s    best!
[ 2000/3000]  66.7%   train=1.4604   val=1.4081   elapsed=1472s   ETA=735s    best!
[ 2200/3000]  73.3%   train=1.4113   val=1.3857   elapsed=1653s   ETA=600s    best!
[ 2400/3000]  80.0%   train=1.3923   val=1.3725   elapsed=1820s   ETA=454s    best!
[ 2600/3000]  86.7%   train=1.3501   val=1.3446   elapsed=1998s   ETA=307s    best!
[ 2800/3000]  93.3%   train=1.3336   val=1.3334   elapsed=2174s   ETA=154s    best!
[ 2999/3000] 100.0%   train=1.3191   val=1.3145   elapsed=2363s   ETA=0s      best!

[DONE] Training finished in 2364.1s (39.4 min)
[DONE] Best val loss: 1.3145
[SAVE] Best weights saved to best_model.pt
```

---

## Head-to-Head Comparison

| Metric | Run 1 — CPU Laptop | Run 2 — GPU Colab | Run 3 — Tesla T4 ⭐ |
|---|---|---|---|
| **Device** | AMD Ryzen 5 CPU | CUDA GPU (Colab) | Tesla T4 (CUDA 13.0) |
| **Parameters** | 0.82M | 10.82M | **1.99M** |
| **Architecture** | 4L × 4H × 128d | 6L × 6H × 384d | 4L × 4H × 200d |
| **Dataset size** | 201,570 chars | 88,406,739 chars | ~31.4M chars |
| **Vocab size** | 28 | 110 | **100** |
| **Block size** | 128 tokens | 256 tokens | 128 tokens |
| **Batch size** | 16 | 64 | 64 |
| **Training steps** | 3,000 | 5,000 | 5,000 |
| **Training time** | 39.4 min | 61.3 min | **6.1 min** |
| **Best val loss** | 1.3145 | **0.7176** | 0.9250 |
| **Overfitting** | None | None | None |
| **Still improving at end?** | Yes | Yes | Yes |

> **Key insight on Run 3:** A 1.99M parameter model on a Tesla T4 reached val loss 0.9250 in just 6.1 minutes — faster than any previous run by a large margin. This shows GPU acceleration pays off even for small models. Run 2 still holds the best quality due to its larger size and more data, but Run 3 shows what efficient GPU use looks like.

---

## Model Output Comparison

### Run 2 — GPU (10.82M params, val loss 0.7176)

```
Upon a time, there were two friends, Jack and Tom. They had a cold doll in
the sunshine.

One day, Jack saw that he was universed. He used the sky at past it to march
around the garden. He had a small ball on his face. He felt dizzy and wanted
to share his happy with them.

Nack knew it was feeling important to his passion in their rooms. He knew
that night, he had never seen a small boy just soon could drink.

He kept helping her passion and seing this boy. As he kept walking, he saw
a girl.
```

### Run 1 — CPU (0.82M params, val loss 1.3145)

```
when years me told be found a big ea reak abig driendly they named not she
rabbit smiled by aded he what in again
one smiled the mushrought boy
one day and was arroom him that she rabbing animals the dreezed at neard had
to there man owl them with o box and said you s mom that je animable went her
somethings of they ballike i wanted a big taught jill hone was and
he rabbit to havin after the but help and nelpft but it was surpring take to
```

### Output Quality Analysis

| Quality Dimension | Run 1 (CPU, 0.82M) | Run 2 (GPU, 10.82M) | Run 3 (T4, 1.99M) |
|---|---|---|---|
| **Sentence structure** | Fragmented | Full sentences | Partial sentences |
| **Story arc** | Weak | Clear narrative flow | Basic flow |
| **Character names** | Inconsistent | Consistent | Moderate |
| **Spelling** | Many errors | Mostly correct | Mostly correct |
| **Word spacing** | Mostly correct | Correct | Correct |
| **Coherence** | Low | Moderate | Low-moderate |
| **Story phrases** | Partial | Natural | Present |
| **Paragraph breaks** | None | Present | Partial |

---

## Loss Curve Analysis

All three runs showed the same characteristic loss curve shape:

```
Phase 1 — Rapid Drop (0–20% of training):
  Run 1 (CPU):     3.30 → 1.70   (model learns basic structure fast)
  Run 2 (Colab):   4.92 → 1.03   (steeper — larger model, more to learn)
  Run 3 (T4):      4.62 → 1.25   (fast drop in just seconds on GPU)

Phase 2 — Steady Descent (20–80%):
  Run 1:   1.70 → 1.39
  Run 2:   1.03 → 0.74
  Run 3:   1.25 → 0.96   (consistent improvement throughout)

Phase 3 — Diminishing Returns (80–100%):
  Run 1:   1.39 → 1.31
  Run 2:   0.74 → 0.72
  Run 3:   0.96 → 0.93   (still falling — more steps would help)
```

All three models were **still improving at the final checkpoint**. None hit a plateau. This means more training steps would reduce loss further in every run.

---

## Overfitting Analysis

**No run showed any overfitting.**

In all three cases, val loss tracked train loss closely and improved monotonically across most checkpoints.

```
Healthy training (all runs):
  train loss ↓ and val loss ↓ together  →  model is generalizing

Overfitting would look like:
  train loss ↓ but val loss ↑           →  model is memorizing
```

The train/val gap at the end of each run:

| Run | Final Train | Final Val | Gap |
|---|---|---|---|
| Run 1 (CPU, 0.82M) | 1.3191 | 1.3145 | 0.0046 |
| Run 2 (Colab, 10.82M) | 0.7259 | 0.7176 | 0.0083 |
| Run 3 (T4, 1.99M) | 0.9307 | 0.9250 | 0.0057 |

All gaps are small and healthy. Run 3 sits between Run 1 and Run 2, which matches its middle-sized architecture.

---

## Scaling Laws — And Where Your Model Sits

<img width="1029" height="705" alt="Screenshot 2026-03-17 171921" src="https://github.com/user-attachments/assets/69b2c840-14f2-4cd1-b4b4-662abda569ff" />

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

### Where All Three Runs Sit

| Model | Parameters | Training Data | Optimal Data (20× rule) | Data Coverage |
|---|---|---|---|---|
| Run 1 — CPU | 0.82M | ~200K tokens | ~16.4M tokens | **1.2%** |
| Run 3 — T4 | 1.99M | ~28.3M tokens | ~39.8M tokens | **71.1%** |
| Run 2 — Colab | 10.82M | ~79.6M tokens | ~216M tokens | **36.8%** |
| GPT-2 Small | 117M | ~40B tokens | ~2.3B tokens | ~1700% |
| GPT-3 | 175B | ~600B tokens | ~3.5T tokens | ~17% |

> **Run 3 stands out here.** At 71.1% of optimal data coverage it is the best-balanced run so far — the model size and dataset size are close to the ideal Chinchilla ratio. Run 2 has the most parameters but sits at only 36.8% coverage, meaning it would benefit more from additional data than additional capacity.

### Full Model Landscape

```
Model                    Parameters    Data               Val Loss    Output Quality
──────────────────────────────────────────────────────────────────────────────────────
Run 1 — CPU (this repo)  0.82M         ~200K tokens       1.3145      Word fragments
Run 3 — T4  (this repo)  1.99M         ~28.3M tokens      0.9250      Basic sentences  ← Efficient run
Run 2 — Colab (this repo)10.82M        ~79.6M tokens      0.7176      Story sentences  ← Best quality
GPT-2 Small              117M          ~40B tokens        ~3.0*       Coherent English
GPT-2 Large              774M          ~40B tokens        ~2.5*       Strong English
GPT-3                    175B          ~600B tokens       —           Near-human text

* GPT-2 losses are on a different (larger) vocabulary and not directly comparable.
```

### What This Tells Us

Run 3 proves that a small, well-balanced model on a fast GPU can converge in minutes. The next logical steps across any run:

```
1. More training steps  →  all three were still falling at the final checkpoint
2. More data            →  Run 3 is closest to optimal ratio; Run 2 benefits most
3. Larger model         →  only worth it once data coverage is above 50%
```

---

## How Weights Produce Output

After training, the model is frozen. The weights file (`best_model.pt`) contains all the numbers that encode everything the model learned from your children's stories.

**The generation loop:**

```
Step 1 — Start with a seed token (start of text)
              ↓
Step 2 — Feed it through all transformer layers
         Each layer does matrix multiplications
         using the saved weight numbers
              ↓
Step 3 — Output is N numbers (one per vocab character)
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

The sampling step picks randomly from the probabilities. Same weights, different random draws = different output each time. To get deterministic output, add `torch.manual_seed(42)` before generation.

---

## Known Limitations

- **Character-level only** — the model learns characters not words, so it cannot spell reliably or track meaning across sentences
- **Output will not be fully coherent** — story fragments are recognizable in Run 2 but still logically drift across paragraphs
- **All models undertrained** — val loss was still falling at the final checkpoint in all three runs; more iterations would help
- **Run 2 still at ~37% optimal data** — a larger story dataset would meaningfully improve output quality
- **No memory between runs** — each generation starts from scratch with no prior context

---

## Real Model Output (Run 1 — CPU)

```
when years me told be found a big ea reak abig driendly they named not she
rabbit smiled by aded he what in again
one smiled the mushrought boy
one day and was arroom him that she rabbing animals the dreezed at neard had
to there man owl them with o box and said you s mom that je animable went her
```

## Real Model Output (Run 2 — GPU Colab)

```
Upon a time, there were two friends, Jack and Tom. They had a cold doll in
the sunshine.

One day, Jack saw that he was universed. He used the sky at past it to march
around the garden. He had a small ball on his face. He felt dizzy and wanted
to share his happy with them.

Nack knew it was feeling important to his passion in their rooms. He knew
that night, he had never seen a small boy just soon could drink.

```
## Real Model Output (Run 3 — GPU )

```
Timmy and elsed him to tell being jumping things. They were tired and making some pinkets and help paper me. They had to see them, drain and ran ar her mommy. They also fast with the stretch and sacks the changer. They play and them together or day. They tlike to need to stay and cut fun and have to catch him. But the bird is pretty. You have to make your legs and it's some people truck in it."

Lily's truck laughed and saw a rock. She said, "You can't here some wet sicks. You have something new favorite toys, I do yours. All of fun!" From that day on, Lily always callimbed the slide, and Tom were playing with the surprise, loved to play in the park. One day, they went to the park with most in the  bathting to girl and dinner. One day, the family went off another floor and quickly made Jack the toys far away. 

In a weath came and dancelet every day out for righting. It was a lot of big ball towers and eggs and make him lots of them. The man is a perfect of the bad lettersser on him. 
```

---

*Built with PyTorch. — [https://github.com/Eamon2009/Transformer-language-model]*
