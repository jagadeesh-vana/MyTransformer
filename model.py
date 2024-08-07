
import torch
import random
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):

    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.keys = nn.Linear(n_embd, head_size, bias=False)
        self.values = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape # (B, T, n_embd)
        q = self.query(x) # (B, T, n_embd/num_heads)
        k = self.keys(x) # (B, T, n_embd/num_heads)
        v = self.values(x) # # (B, T, n_embd/num_heads)
        wei = q @ k.transpose(-2,-1) *C **-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v # (B, T, n_embd/num_heads)
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x ):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, n_embd)
        out = self.dropout(self.proj(out)) # (B, T, n_embd)
        return out

class FeedForward(nn.Module):
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.2),
        )
        
    def forward(self, x):
        return self.net(x) # (B, T, n_embd)
        
class Block(nn.Module):
    
    def __init__(self, n_embd, block_size):
        super().__init__()
        self.mmsa = MultiHeadAttention(4, n_embd//4, n_embd, block_size)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.mmsa(self.ln1(x)) # (B, T, n_embd)
        x = x + self.ff(self.ln2(x)) # (B, T, n_embd)
        return x
        

class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, n_layer):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) 
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape  # (B, T)
        token_embeddings = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_embeddings = self.position_embedding_table(torch.arange(T)) # (T, n_embd)
        x = token_embeddings + pos_embeddings # (B, T, n_embd)
        x = self.blocks(x) # (B, T, n_embd)
        x = self.ln_f(x) # (B,T,n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            targets = targets.view(B*T)
            logits = logits.view(B*T, C)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, block_size, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx