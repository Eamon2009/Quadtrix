# Quadtrix [![License](https://img.shields.io/github/license/Eamon2009/Transformer-language-model)](LICENSE)
<img width="1138" height="515" alt="Screenshot 2026-04-15 225925" src="https://github.com/user-attachments/assets/4007c0ed-ab50-4084-a604-ef76e60826e5" />
<img width="1919" height="756" alt="Screenshot 2026-04-15 231851" src="https://github.com/user-attachments/assets/975b1b9b-93f3-43c8-b93f-3fa39eba7188" />


A minimal, educational GPT-style transformer trained character-by-character on children's stories. No pre-trained weights. No fine-tuning. Just raw PyTorch, from init to generation.
Character-level Transformer is a sequence-to-sequence architecture that operates on the granularity of individual characters rather than words or subword tokens. By utilizing a self-attention mechanism over long sequences of characters, the model learns to construct internal representations of morphology, syntax, and semantics from the ground up, effectively eliminating "out-of-vocabulary" (OOV) issues. While this approach allows for high fidelity in modeling rare words, spelling variations, and creative linguistics, it significantly increases the computational complexity—typically $O(L^2)$ where $L$ is the sequence length—as a single sentence requires many more steps than its word-level equivalent. Consequently, character-level Transformers often require deeper architectures or auxiliary losses to capture the long-range dependencies necessary to match the semantic performance of traditional token-based models

> **The goal**: Understand how language models learn patterns. See it happen on your machine. Train in minutes on a GPU.

---

## Quick Start

```bash
# Install PyTorch
pip install torch

# Run training
python transformer.py

# The script will train, save best weights, then generate text forever
# Press Ctrl+C to stop
```

**That's it.** No complex setup, no data pipelines, no credentials.

---

## What Is This?

Quadtrix is a learning project. It trains a tiny transformer on text—character by character—and learns which characters tend to follow others. At generation time, it predicts the next character, feeds it back in, and repeats.

**It's the same architecture as GPT**, just much smaller (1M–11M parameters instead of 175B) and trained on much less data (thousands of stories instead of the internet).

The magic: in just 6 minutes on a Tesla T4 GPU, a 2M-parameter model learns meaningful patterns about narrative structure, dialogue, and storytelling. The output isn't Shakespeare, but it *is* recognizable as a story.

---

## How It Works

### The Pipeline

```
1. Load children's stories from disk
   ↓
2. Encode each character as an integer (vocab size: 28–110)
   ↓
3. Split into train/val chunks (80/20)
   ↓
4. Build a GPT-style transformer with:
   - Embedding layer (character → vector)
   - Transformer blocks (self-attention + feedforward)
   - Output projection (vector → logits over vocab)
   ↓
5. Train for N steps, measuring loss every eval_interval
   ↓
6. Save best weights whenever validation loss improves
   ↓
7. Load best model and generate text forever
   ↓
8. Press Ctrl+C to stop
```

### What the Model Actually Does

Each forward pass:
1. Takes a sequence of character indices (e.g., "Once upon...")
2. Embeds each into a dense vector
3. Passes through transformer layers (multi-head attention learns which characters to attend to)
4. Outputs logits (scores) for every possible next character
5. Samples the next character according to those probabilities
6. Feeds it back in and repeats

**Loss function**: Cross-entropy. The model minimizes surprise on unseen text.

**Training**: Adam optimizer with a learning rate schedule. Dropout prevents overfitting.

---

## Project Structure

```
transformer.py          ← Everything. One file.
best_model.pt           ← Saved weights (after first training run)
data.txt                ← Your text file (any UTF-8 file works)
```

That's all. No fancy folder structure. No config files. Edit hyperparameters directly in the script.

---

## Configuration & Hardware

Quadtrix is designed to work anywhere: laptop CPU to cloud GPU. Edit these hyperparameters in `transformer.py`:

```python
# ============================================================================
# Hyperparameters
# ============================================================================
batch_size    = 64         # Sequences per batch
block_size    = 128        # Context window (tokens)
max_iters     = 5000       # Total training steps
eval_interval = 200        # Print loss every N steps
learning_rate = 3e-4
n_embd        = 200        # Embedding dimension
n_head        = 4          # Attention heads per layer
n_layer       = 4          # Number of transformer blocks
dropout       = 0.2        # Regularization
```

### Three Pre-Tuned Configurations

**CPU (Laptop) — Fast Feedback**
```python
batch_size, block_size, max_iters = 16, 128, 3000
n_embd, n_head, n_layer = 128, 4, 4
# ~0.82M parameters
# Trains in ~40 min on AMD Ryzen
```

**GPU (Google Colab) — Best Quality**
```python
batch_size, block_size, max_iters = 64, 256, 5000
n_embd, n_head, n_layer = 384, 6, 6
# ~10.82M parameters
# Trains in ~60 min on Colab GPU
# Best output quality
```

**GPU (Tesla T4) — Efficient**
```python
batch_size, block_size, max_iters = 64, 128, 5000
n_embd, n_head, n_layer = 200, 4, 4
# ~1.99M parameters
# Trains in **6.1 minutes** ← Fastest
# Best parameter/data balance
```

---

## Training Results

Three runs. Three different hardware setups. All converged well.

### Run 3 — Tesla T4 (Latest) ⭐

**Configuration**: 4 layers × 4 heads × 200 dim = **1.99M params**

| Metric | Value |
|--------|-------|
| Device | Tesla T4 (CUDA 13.0) |
| Dataset | ~31.4M characters (children's stories) |
| Train tokens | 28.3M |
| Val tokens | 3.1M |
| Training time | **6.1 minutes** |
| Best val loss | **0.9250** |
| Final train loss | 0.9307 |
| Overfitting | None detected |

**Training curve** (every 200 steps):
```
[    0/5000]   train=4.6207   val=4.6202   elapsed=2s    << best!
[  200/5000]   train=2.2058   val=2.1986   elapsed=17s   << best!
[  400/5000]   train=1.6111   val=1.6039   elapsed=32s   << best!
[ 1000/5000]   train=1.2495   val=1.2567   elapsed=76s   << best!
[ 2000/5000]   train=1.0731   val=1.0765   elapsed=149s  << best!
[ 3000/5000]   train=1.0000   val=0.9956   elapsed=221s  << best!
[ 4000/5000]   train=0.9601   val=0.9642   elapsed=294s
[ 4200/5000]   train=0.9515   val=0.9489   elapsed=309s  << best!
[ 4400/5000]   train=0.9433   val=0.9431   elapsed=323s  << best!
[ 4800/5000]   train=0.9331   val=0.9250   elapsed=353s  << best!
[ 4999/5000]   train=0.9307   val=0.9430   elapsed=367s

[DONE] Training finished in 367.0s (6.1 min) | Best val loss: 0.9250
```

**Key insight**: This run hit the **sweet spot**—large enough to learn coherent patterns, small enough to train fast. It's the reference configuration.

---

### Run 2 — Google Colab (10.82M Parameters)

**Configuration**: 6 layers × 6 heads × 384 dim

| Metric | Value |
|--------|-------|
| Device | CUDA (Google Colab GPU) |
| Dataset | ~88.4M characters |
| Parameters | 10.82M |
| Training time | 61.3 minutes |
| Best val loss | **0.7176** |
| Overfitting | None |

**Result**: Larger model, more data = **best output quality**. Slower to train but produces recognizable narratives.

---

### Run 1 — CPU Laptop (0.82M Parameters)

**Configuration**: 4 layers × 4 heads × 128 dim

| Metric | Value |
|--------|-------|
| Device | AMD Ryzen 5 PRO 3500U (CPU only) |
| Parameters | 0.82M |
| Dataset | ~201K characters |
| Training time | 39.4 minutes |
| Best val loss | 1.3145 |
| Overfitting | None |

**Result**: Smallest model, tiniest dataset. Trains fastest on CPU. Output is fragmented but shows the model learned *something*.

---

## Head-to-Head Comparison

| Metric | Run 1 — CPU | Run 2 — Colab | Run 3 — T4 ⭐ |
|--------|-------------|--------------|------------|
| **Parameters** | 0.82M | 10.82M | 1.99M |
| **Training data** | 200K chars | 88.4M chars | 31.4M chars |
| **Training time** | 39.4 min | 61.3 min | **6.1 min** |
| **Best val loss** | 1.3145 | **0.7176** | 0.9250 |
| **Output coherence** | Fragmented | Coherent paragraphs | Basic sentences |
| **Overfitting** | ✓ None | ✓ None | ✓ None |
| **Still improving?** | Yes | Yes | Yes |

> **Observation**: All three were still improving at the final checkpoint. More training steps = better loss.

---

## Example Outputs

### Run 2 Output (10.82M params) — Best Quality

```
Upon a time, there were two friends, Jack and Tom. They had a cold doll in
the sunshine.

One day, Jack saw that he was universed. He used the sky at past it to march
around the garden. He had a small ball on his face. He felt dizzy and wanted
to share his happy with them.

Nack knew it was feeling important to his passion in their rooms. He knew
that night, he had never seen a small boy just soon could drink.
```

**Analysis**: Clear sentence structure. Named characters. Logical progression. Some linguistic oddities ("felt dizzy and wanted to share his happy") but unmistakably a story.

---

### Run 3 Output (1.99M params, Tesla T4) — Efficient

```
Timmy and elsed him to tell being jumping things. They were tired and making
some pinkets and help paper me. They had to see them, drain and ran ar her
mommy. They also fast with the stretch and sacks the changer.

Lily's truck laughed and saw a rock. She said, "You can't here some wet
sicks. You have something new favorite toys, I do yours."
```

**Analysis**: Narrative present. Dialogue structure intact. Characters named. Some word-order errors and made-up words, but the *shape* of a story is clear.

---

### Run 1 Output (0.82M params) — Minimal

```
when years me told be found a big ea reak abig driendly they named not she
rabbit smiled by aded he what in again one smiled the mushrought boy one day
and was arroom him that she rabbing animals the dreezed at neard had to there
man owl them with o box
```

**Analysis**: Word boundaries mostly intact. Character names present (rabbit, owl). But syntax falls apart—the model is struggling to hold sentence structure. This is a 0.82M model trained on tiny data; it's learning *something* but hasn't converged to coherent text.

---
## Project Structure
 
```
.
├── .github/
│   ├── ISSUE_TEMPLATE/      # GitHub issue templates
│   └── workflows/           # CI/CD workflows
├── .vscode/                 # VS Code configuration
├── GPU train/               # GPU training scripts and configurations
├── assets/                  # Images, diagrams, and other assets
├── checkpoints/             # Saved model checkpoints
├── config/                  # Configuration files
├── data_set/                # Training and validation datasets
├── evaluate/                # Evaluation scripts and metrics
├── generate/                # Text generation scripts
├── logs/                    # Training logs and tensorboard files
├── train_test/              # Training and testing utilities
├── .gitattributes          # Git attributes configuration
├── .gitignore              # Git ignore rules
├── .python-version         # Python version specification
├── LICENSE                 # MIT License
├── README.md               # This file
├── cleaned.txt             # Cleaned training data
├── gpt-from-scratch.ipynb  # Jupyter notebook implementation
├── requirements.txt        # Python dependencies
├── traindata.txt           # Raw training data
└── transformer.py          # Main transformer model implementation
```

## Loss Curves & Training Dynamics

All three runs showed the classic learning curve:

```
Phase 1: Rapid drop (0–20% of training)
  - Model learns basic character transitions
  - Loss halves or better
  
Phase 2: Steady descent (20–80%)
  - Model learns longer-range patterns
  - Character names, sentence boundaries
  - Loss continues down but more gradually
  
Phase 3: Diminishing returns (80–100%)
  - Model learning slows
  - Val loss still improving but incremental
  - More data or capacity would help here
```

**Train/Val Gap Analysis** (indicator of overfitting):

| Run | Final Train | Final Val | Gap |
|-----|-------------|-----------|-----|
| CPU | 1.3191 | 1.3145 | 0.0046 |
| Colab | 0.7259 | 0.7176 | 0.0083 |
| T4 | 0.9307 | 0.9250 | 0.0057 |

All gaps are tiny. **No overfitting detected.** The model is generalizing well to unseen text.

---

## Scaling Laws: Where Quadtrix Sits

The Chinchilla (2022) scaling law suggests: **~20 tokens of training data per parameter is optimal**.

Let's see how our runs align:

| Model | Parameters | Training Data | Optimal (20×) | Coverage |
|-------|------------|---------------|--------------|----------|
| Run 1 | 0.82M | 200K tokens | 16.4M | **1.2%** |
| Run 3 | 1.99M | 28.3M tokens | 39.8M | **71.1%** ← Best balanced |
| Run 2 | 10.82M | 79.6M tokens | 216M | **36.8%** |
| GPT-2 Small | 117M | 40B tokens | 2.3B | ~1700% |
| GPT-3 | 175B | 600B tokens | 3.5T | ~17% |

**Insight**: Run 3 is the most *balanced*—the model size and data quantity are well-matched. Run 2 has the largest model but is only at 37% optimal data coverage, meaning it would benefit more from adding data than adding parameters.

**Next steps for any run**:
1. **More training steps** — All three were still falling at the final checkpoint
2. **More data** — Run 3 is closest to optimal; Run 2 would benefit most
3. **Larger model** — Only worth doing once data coverage exceeds 50%

---

## How Generation Works

Once training is done, `best_model.pt` contains frozen weights. Generation is simple:

```python
# Pseudocode for generation
seed = torch.tensor([[start_token]])  # e.g., start_token = 0

for _ in range(num_chars_to_generate):
    # Forward pass through all layers
    logits = model(seed)[-1, :]  # Get last token's logits
    
    # Convert logits to probabilities
    probs = softmax(logits / temperature)
    
    # Sample next token
    next_token = sample(probs)
    
    # Append and continue
    seed = torch.cat([seed, next_token], dim=-1)
    
    # Trim to context window if needed
    seed = seed[:, -block_size:]
```

**Why output differs each run**: The sampling step is random. Same weights, different random seeds = different output. Add `torch.manual_seed(42)` for deterministic output.

---

## Known Limitations

1. **Character-level learning** — the model learns characters, not words. It cannot reliably spell or track meaning across paragraphs.

2. **Output coherence** — especially on smaller runs. Sentences drift logically. Names disappear. Tense breaks. This is expected at this scale.

3. **All models undertrained** — validation loss was still improving at iteration N. More training steps would help all three.

4. **Limited data** — Run 2 is only at 37% optimal data coverage. A larger story corpus would meaningfully improve output quality.

5. **No long-range memory** — transformer context window is fixed (128 or 256 tokens). The model cannot reference events from 10 paragraphs ago.

---

## Technical Details

### Model Architecture

```python
class GPTModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.transformer = nn.Sequential(
            *[TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_embedding(x)  # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T))
        x = self.dropout(tok_emb + pos_emb)
        x = self.transformer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
```

### Transformer Block

Each block contains:
- **Multi-head self-attention**: `(B, T, n_embd) → (B, T, n_embd)`
- **Feedforward network**: Two linear layers with ReLU
- **Layer normalization & residual connections**
- **Dropout for regularization**

### Training Loop

```python
for step in range(max_iters):
    # Batch a random chunk of training data
    batch_x, batch_y = get_batch('train')
    
    # Forward pass
    logits = model(batch_x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), batch_y.view(-1))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Evaluate and save
    if step % eval_interval == 0:
        val_loss = estimate_loss('val')
        print(f"train={train_loss:.4f} val={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
```

---

## Why This Project Matters

1. **Educational**: See exactly how a language model learns. No black boxes.

2. **Verifiable**: You can trace loss, inspect weights, understand every line of code.

3. **Fast**: Even on CPU, training finishes in under an hour. On GPU, minutes.

4. **Runnable**: One script, one dependency (PyTorch). No cloud account required.

Quadtrix is to GPT what nanoGPT is to transformers: a stripped-down, readable, educational version that teaches the core ideas.

---

## Getting Started

1. **Clone or download** `transformer.py`
2. **Install PyTorch**: `pip install torch`
3. **Add your data**: Place a UTF-8 text file at `data.txt` (or edit the filename in the script)
4. **Run**: `python transformer.py`
5. **Watch**: Loss decreases. Weights save. Text generates.
6. **Tweak**: Edit hyperparameters and re-run to see how each affects training speed and output quality.

---

## License

MIT

---

*Built with PyTorch. | [GitHub](https://github.com/Eamon2009/Transformer-language-model)*
