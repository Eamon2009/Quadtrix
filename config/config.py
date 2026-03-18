import torch

# training
batch_size = 4
block_size = 18
train_split = 0.9

# reproducibility
seed = 1337

# device
device = "cpu" if torch.cuda.is_available() else "cuda"