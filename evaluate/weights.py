import torch
import sys
sys.path.append(r'C:\Users\Admin\Documents\GitHub\Transformer-language-model')

from transformer import GPTLanguageModel

vocab_size = 28

model = GPTLanguageModel(vocab_size)
model.load_state_dict(torch.load(
    r'C:\Users\Admin\Documents\GitHub\Transformer-language-model\best_model.pt',
    map_location=torch.device('cpu')
))
model.eval()
print("Model loaded successfully!")