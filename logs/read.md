# Logs

This folder stores training logs, loss curves, and run history.

## Structure

```
logs/
├── train_loss.csv        ← loss per epoch (easy to plot)
├── train_run_latest.log  ← full console output of latest run
└── run_YYYYMMDD_HHMM.log ← timestamped logs per run
```

## How to Log (add to your transformer.py / train script)

```python
import csv
import os
import logging
from datetime import datetime

# --- Setup ---
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# Console + file logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(f"logs/run_{timestamp}.log"),
        logging.FileHandler("logs/train_run_latest.log", mode="w"),
        logging.StreamHandler()  # still prints to terminal
    ]
)
logger = logging.getLogger(__name__)

# CSV loss tracker
loss_csv = "logs/train_loss.csv"
if not os.path.exists(loss_csv):
    with open(loss_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

def log_epoch(epoch, train_loss, val_loss=None, lr=None):
    logger.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}" +
                (f" | Val Loss: {val_loss:.4f}" if val_loss else "") +
                (f" | LR: {lr:.6f}" if lr else ""))
    with open(loss_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, val_loss or "", lr or ""])
```

## Usage in Training Loop

```python
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, optimizer)
    log_epoch(epoch, train_loss, lr=scheduler.get_last_lr()[0])
    save_checkpoint(model, optimizer, epoch, train_loss)
```

## Plotting Loss Curve

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/train_loss.csv")
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
if "val_loss" in df and df["val_loss"].notna().any():
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GPT Training Loss")
plt.legend()
plt.savefig("logs/loss_curve.png")
plt.show()
```

## Notes

- `.log` files and `loss_curve.png` can be pushed to GitHub (they're small)
- `train_loss.csv` is very useful to track across GPU runs
- Timestamped logs help you compare different training runs