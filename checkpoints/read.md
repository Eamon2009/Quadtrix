# Checkpoints

This folder stores saved model weights during and after training.

## Structure

```
checkpoints/
├── best_model.pt          ← best checkpoint by validation loss
├── latest_model.pt        ← most recent checkpoint
└── epoch_XX_loss_X.XX.pt  ← epoch-specific saves
```

## How to Save (add to your transformer.py / train script)

```python
import torch
import os

def save_checkpoint(model, optimizer, epoch, loss, path="checkpoints/"):
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    # Save latest
    torch.save(checkpoint, os.path.join(path, "latest_model.pt"))

    # Save best if loss improved
    best_path = os.path.join(path, "best_model.pt")
    if not os.path.exists(best_path):
        torch.save(checkpoint, best_path)
    else:
        best = torch.load(best_path)
        if loss < best["loss"]:
            torch.save(checkpoint, best_path)
            print(f" New best model saved at epoch {epoch} with loss {loss:.4f}")

    # Save per epoch (optional)
    torch.save(checkpoint, os.path.join(path, f"epoch_{epoch:03d}_loss_{loss:.4f}.pt"))
```

## How to Load

```python
def load_checkpoint(model, optimizer, path="checkpoints/latest_model.pt"):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f" Loaded checkpoint from epoch {epoch}, loss: {loss:.4f}")
    return model, optimizer, epoch, loss
```

## Notes

- `.pt` files are listed in `.gitignore` — they are too large for GitHub
- Push only this README and `.gitkeep` to keep the folder tracked by git
- For long GPU runs, save every N epochs so you can resume if it crashes