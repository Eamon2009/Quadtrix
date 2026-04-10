import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPTLanguageModel, device, _model_path, _script_path

print("Loading weights...")
model = GPTLanguageModel().to(device)
model.load_state_dict(torch.load(_model_path, map_location=device, weights_only=True))
model.eval()

print("Exporting to TorchScript...")
scripted = torch.jit.script(model)
scripted.save(_script_path)
print(f"Saved TorchScript model to: {_script_path}")