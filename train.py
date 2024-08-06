import torch
import random
import torch.nn as nn
from torch.nn import functional as F

from model import GPTLanguageModel

with open('./input.txt') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for  i, ch in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : ''.join([itos[i] for i in l])
data = torch.tensor(encode(text))

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i+ block_size] for i in idx])
    y = torch.stack([data[i+1: i+ block_size+1] for i in idx])
    return x, y

n_embd = 32
block_size = 8
batch_size = 32
n_layer = 2

model = GPTLanguageModel(vocab_size, n_embd, block_size, n_layer)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

max_iters = 10
eval_interval = 4
eval_iters = 2

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for iter in range(max_iters):
    if iter//eval_interval == 0 or iter == max_iters-1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    x, y = get_batch('train')
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(context, block_size, max_new_tokens=500)[0].tolist()))