import torch
import torch
import torch.nn as nn
from torch.nn import functional as F
import re

with open("traindata.txt",'r',encoding='utf-8') as f:
       text=f.read()
chars=list(set(text))
vocab_size=len(chars)
print(''.join(chars))
print(vocab_size)

with open("traindata.txt", "r", encoding="utf-8") as f:
    text = f.read()

# keep alphabets, spaces, and newline
text = re.sub(r"[^A-Za-z\s\n]", " ", text)

# clean extra spaces but keep line structure
text = re.sub(r"[ ]+", " ", text)

# convert to lowercase
text = text.lower()

with open("cleaned.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("cleaning complete")

with open("cleaned.txt",'r',encoding='utf-8') as f:
       text2=f.read()

chars2=sorted(list(set(text2)))
vocab_size2=len(chars2)
print(''.join(chars2))
print(vocab_size2)

stri = {ch:i for i, ch in enumerate(chars2)}
it = {i:ch for i, ch in enumerate(chars2)}

encode = lambda s: [stri[c] for c in s]# a function that take a string triverse it and return a integer list
decode = lambda l: ''.join([it[i] for i in l])# a function that convert the 

print(encode("eamon"))
print(decode(encode("eamon")))
data=torch.tensor(encode(text2),dtype=torch.long)
print(data.dtype,data.shape)
# now split the data into train data and validation data
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]
block_size=18
train_data[:block_size+1]

a=train_data[:block_size]
b=train_data[1:block_size+1]
for t in range(block_size):
       context=a[:t+1]
       target=b[t]
       print(f"Input:{context},target:{target}")
batch_size=4
block_size=18
torch.manual_seed(1337)
def get_batch(split):
       if split=='train':
              data=train_data
       else:
              data=val_data
       ix = torch.randint(len(data) - block_size, (batch_size,))
       x=torch.stack([data[i:i+block_size]for i in ix])
       y=torch.stack([data[i+1:i+block_size+1]for i in ix])
       return x,y

xb,yb=get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

for b in range(batch_size):
       for t in range(block_size):
              context=xb[b,:t+1]
              target=yb[b,t]
              print(f"context:{context} and target:{target} ")

torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size2)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
