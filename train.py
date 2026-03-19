import torch
import torch.nn as nn
import re
from torch.nn import functional as F
import time
from config.config import *

start = time.time()

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu' # my case it's the cpu
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()
text = re.sub(r"[^A-Za-z\s\n]", " ", text)  # keep alphabets, spaces, and newline
text = re.sub(r"[ ]+", " ", text)           # clean extra spaces but keep line structure
text = text.lower()                         # convert to lowercase
with open(cleaned_path, "w", encoding="utf-8") as f:
    f.write(text)
with open(cleaned_path, 'r', encoding='utf-8') as f:
    text2 = f.read()

chars2 = sorted(list(set(text2)))
vocab_size = len(chars2)
stri = {ch: i for i, ch in enumerate(chars2)}
it   = {i: ch for i, ch in enumerate(chars2)}
encode = lambda s: [stri[c] for c in s]         # a function that take a string triverse it and return a integer list
decode = lambda l: ''.join([it[i] for i in l])  # a function that convert the

data = torch.tensor(encode(text2), dtype=torch.long)
# print(data.dtype, data.shape)

# now split the data into train data and validation data
n = int(train_split * len(data))
train_data = data[:n]
val_data   = data[n:]

train_data[:block_size + 1]

a = train_data[:block_size]
b = train_data[1:block_size + 1]
for t in range(block_size):
    context = a[:t + 1]
    target  = b[t]
    # print(f"Input:{context},target:{target}")

torch.manual_seed(seed)

def get_batch(split):
    if split == 'train':
        data = train_data
    else:
        data = val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([data[i:i + block_size]     for i in ix])
    y  = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y
xb, yb = get_batch('train')
for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t + 1]
        target  = yb[b, t]
        # print(f"context:{context} and target:{target} ")

torch.manual_seed(seed)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)  # each token directly reads off the logits for the next token from a lookup table

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss    = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)                             # get the predictions
            logits = logits[:, -1, :]                            # becomes (B, C)
            probs = F.softmax(logits, dim=-1)                    # (B, C) softmax to get probabilities
            idx_next = torch.multinomial(probs, num_samples=1)   # sample from the distribution(B, 1)
            idx = torch.cat((idx, idx_next), dim=1)              # append sampled index to the running sequence(B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out
def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


# Adding a Py torch optimizer 
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
# Training loop
batch_size = 32
for steps in range(10000):
    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


model = BigramLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')   # print the number of parameters in the model

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=1000)[0].tolist()))
end = time.time()
print("The time taken :", end - start)