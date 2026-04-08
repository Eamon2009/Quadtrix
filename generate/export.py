import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPTLanguageModel, device

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_model_path   = os.path.join(_project_root, 'best_model.pt')
_script_path  = os.path.join(_project_root, 'best_model_script.pt')

print("Loading weights...")
model = GPTLanguageModel().to(device)
model.load_state_dict(torch.load(_model_path, map_location=device, weights_only=True))
model.eval()

print("Exporting to TorchScript...")
scripted = torch.jit.script(model)

scripted.save(_script_path)
print(f"Saved TorchScript model to: {_script_path}")