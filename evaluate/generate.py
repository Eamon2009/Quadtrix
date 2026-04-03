import torch
import os
from GPU_test import GPTLanguageModel

# 1. Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_PATH = 'best_model.pt'
DATA_PATH = 'data.txt' # Your original 110-vocab file

# 2. Reconstruct Vocabulary from original data
if not os.path.exists(DATA_PATH):
    print(f"Error: Need {DATA_PATH} to match the character mappings!")
    exit()

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
it = {i: ch for i, ch in enumerate(chars)}
decode = lambda l: ''.join([it[i] for i in l])

print(f"Detected Vocab Size: {vocab_size}")

# 3. Build and Load Model
model = GPTLanguageModel(vocab_size).to(device)

if os.path.exists(SAVE_PATH):
    # This should now work without the size mismatch error!
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    model.eval()
    print(f"Weights loaded successfully from {SAVE_PATH}")
else:
    print(f"Error: {SAVE_PATH} not found.")
    exit()

# 4. Generate
print("\n" + "="*60)
print("  GENERATED OUTPUT")
print("="*60 + "\n")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
with torch.no_grad():
    output = model.generate(context, max_new_tokens=1000)
    print(decode(output[0].tolist()))

print("\n" + "="*60)