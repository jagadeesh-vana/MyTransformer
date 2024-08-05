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

def split(split):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i+ block_size] for i in idx])
    y = torch.stack([data[i+1: i+ block_size+1] for i in idx])
    return x, y

n_embd = 32
block_size = 8
batch_size = 32

m = GPTLanguageModel(vocab_size, n_embd, block_size)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for _ in range(10):
    x, y = split('train')
    logits, loss = m(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(loss)