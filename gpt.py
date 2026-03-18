from gpt import GPT
from train import get_batch
from config import *

def train():
    model = GPT()
    model.train()

    for step in range(max_iters):
        xb, yb = get_batch()

        logits, loss = model(xb, yb)

        loss.backward()

        if step % eval_interval == 0:
            print(f"step {step} loss {loss.item()}")

if __name__ == "__main__":
    train()