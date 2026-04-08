import torch
import numpy as np
import os
import sys
import struct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPTLanguageModel, device, _model_path, vocab_size, block_size, n_embd, n_head, n_layer, chars

_project_root  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_weights_path  = os.path.join(_project_root, 'weights.bin')
_vocab_path    = os.path.join(_project_root, 'vocab.bin')

print("Loading weights...")
model = GPTLanguageModel().to('cpu')
model.load_state_dict(torch.load(_model_path, map_location='cpu', weights_only=True))
model.eval()
print("Exporting vocab...")
with open(_vocab_path, 'wb') as f:
    # Write vocab size, then each char as a byte
    f.write(struct.pack('i', vocab_size))
    for ch in chars:
        f.write(struct.pack('B', ord(ch)))
print(f"Vocab saved to: {_vocab_path}  ({vocab_size} chars)")
print("Exporting weights...")

def write_tensor(f, tensor):
    data = tensor.detach().cpu().float().numpy()
    f.write(struct.pack('i', data.ndim))
    for s in data.shape:
        f.write(struct.pack('i', s))
    f.write(data.tobytes())

with open(_weights_path, 'wb') as f:
    # Write config header
    f.write(struct.pack('iiiii', vocab_size, block_size, n_embd, n_head, n_layer))

    # Token + position embeddings
    write_tensor(f, model.token_embedding_table.weight)
    write_tensor(f, model.position_embedding_table.weight)

    # Each block
    for block in model.blocks:
        for head in block.sa.heads:
            write_tensor(f, head.key.weight)
            write_tensor(f, head.query.weight)
            write_tensor(f, head.value.weight)
        write_tensor(f, block.sa.proj.weight)
        write_tensor(f, block.sa.proj.bias)
        write_tensor(f, block.ffwd.net[0].weight)
        write_tensor(f, block.ffwd.net[0].bias)
        write_tensor(f, block.ffwd.net[2].weight)
        write_tensor(f, block.ffwd.net[2].bias)
        write_tensor(f, block.ln1.weight)
        write_tensor(f, block.ln1.bias)
        write_tensor(f, block.ln2.weight)
        write_tensor(f, block.ln2.bias)

    write_tensor(f, model.ln_f.weight)
    write_tensor(f, model.ln_f.bias)
    write_tensor(f, model.lm_head.weight)
    write_tensor(f, model.lm_head.bias)

print(f"Weights saved to: {_weights_path}")
print("Done.")