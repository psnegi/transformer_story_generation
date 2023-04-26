import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import tiktoken

# hyper parameters

BATCH_SIZE = 64
BLOCK_SIZE = 128
MAX_ITER = 10000
EVAL_INTERVAL = 500
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200

torch.manual_seed(1337)

with open('input.txt', encoding='utf-8') as f:
  text= f.read()

vocab = sorted(list(set(text)))


stoi = {c :i for i,c in enumerate(vocab)}
itos = {i : c for c,i in stoi.items()}

enc = tiktoken.get_encoding('gpt2')
encoder = enc.encode
decoder = enc.decode
N_VOCAB = enc.n_vocab
#encoder = lambda x: [stoi[i] for i in x]
#decoder = lambda x: ''.join([itos[i] for i in x])

data = torch.tensor(encoder(text), dtype=torch.long)


n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  data = train_data if split == "train" else val_data
  # find multiple random seed 
  ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
  x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
  y = torch.stack([data[i +1:i+BLOCK_SIZE + 1] for i in ix])
  x, y = x.to(DEVICE), y.to(DEVICE)
  return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
      losses = torch.zeros(EVAL_ITERS)
      for i in range(EVAL_ITERS):
        X,y = get_batch(split)
        logits, loss = model(X,y)
        losses[i] = loss.item()
      out[split] = losses.mean()
      
    model.train()
    return out


train_xb, train_yb = get_batch("train")
val_xb, val_yb = get_batch("validation")


torch.manual_seed(1337)

# Single self attension head
class Head(nn.Module):
  def __init__(self, n_embed, head_size, dropout = .5):
    super().__init__()
    self._head_size = head_size
    self._dropout = nn.Dropout(dropout)
    self._K = nn.Linear(n_embed, head_size, bias = False)
    self._Q = nn.Linear(n_embed, head_size, bias = False)

    self._V = nn.Linear(n_embed, head_size, bias=False)
    self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE,BLOCK_SIZE)))    

  def forward(self, x):

    B, T, C = x.shape
    key = self._K(x)# B, T, head_size, x@K
    query = self._Q(x) # B, T, head_size
    # Afinity or weiht beteween token, see how it is learnd in data dependent way instead of constant values for average

    wei = query @ key.transpose(-2, -1) # B, T, T

    wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))# this line is only in decoder block,
    #in encoder block it is ok for whole tokens to talk to  each other, like in sentiment classification

    wei = torch.softmax(wei, dim = -1)
    wei = self._dropout(wei)
    # Don't just aggregate raw x but a linear transformation of them
    x= self._V(x) #  B, T, head_size
    out =wei@x# B, T, T @ B, T, head_size -> B, T, head_size
    return out       

class MutliHeadAttention(nn.Module):
  def __init__(self, n_embed, n_heads, head_size, dropout):
    super().__init__()
    self._heads = nn.ModuleList([Head(n_embed, head_size, dropout) for _ in range(n_heads)])
    self._proj = nn.Linear(n_embed, n_embed)
    self._dropout = nn.Dropout(dropout)

  def forward(self, x):
    x_temp = torch.cat([h(x) for h in self._heads], dim = -1)
    x_temp = self._proj(x_temp)
    x_temp = self._dropout(x_temp)
    return x_temp
    
class MLP(nn.Module):
  def __init__(self, n_embed, dropout = .5):
    super().__init__()
    self._mlp = nn.Sequential(nn.Linear(n_embed, 4*n_embed),
                               nn.ReLU(),
                               nn.Linear(4*n_embed, n_embed), # Projection back(e.g for adding back to input)
                               nn.Dropout(dropout))
  def forward(self, x):
    return self._mlp(x)


class Block(nn.Module):
  # Add self attention and non linearity on learning(computation) 
  def __init__(self, n_embed, n_heads, dropout = .5):
    super().__init__()
    head_size = n_embed // n_heads
    self._attn = MutliHeadAttention(n_embed, n_heads, head_size, dropout)
    self._ffwd = MLP(n_embed, dropout)
    self._ln1 = nn.LayerNorm(n_embed)
    self._ln2 = nn.LayerNorm(n_embed)

  def forward(self, x):
    x = x + self._attn(self._ln1(x)) # Pre norm formulation for layer norm
    x = x + self._ffwd(self._ln2(x))
    return x


class TransformerEncoderModel(torch.nn.Module):
  def __init__(self, n_vocab = N_VOCAB, n_embed = 64, n_layer = 4):
    super().__init__()    
    # This of this embedding as specifying the probability of next character
    self._embedding = nn.Embedding(n_vocab, n_embed)
    self._pos_embedding = nn.Embedding(BLOCK_SIZE, n_embed)
    self._blocks = nn.Sequential(*[Block(n_embed, n_heads=4) for _ in range(n_layer)])
    self._ln = nn.LayerNorm(n_embed) 
    self._lm_head = nn.Linear(n_embed, n_vocab)


    

  def forward(self, x, y = None): # B, T
    B, T = x.shape
    token_embed = self._embedding(x)# B, T, embed dim
    pos_embed = self._pos_embedding(torch.arange(T, device=DEVICE)) # T, embed dim
    x_temp = token_embed + pos_embed
    x_temp = self._blocks(x_temp)
    x_temp = self._ln(x_temp)
    logits = self._lm_head(x_temp) # B, T, C tensor 
    B, T, C = logits.shape
 
    
    #y_one_hot = F.one_hot(y, self._nclasses) # B, T, C tensor.e
    if y is None:
      loss = None
    else:  
      loss = F.cross_entropy(logits.view(B*T, C), y.view(B*T))
    return logits, loss
   

  def generate(self, idx, max_len):
    #get hte prediction, idx : B, T
    # sample from the problabilites in a loop
    #predictions=idx
    #last_idx = idx[:, -1][:, None] # B, 1
    #print(last_idx.shape)
    for i in range(max_len):
      idx_cond = idx[:, -BLOCK_SIZE:]
      logits, _ = self(idx_cond) # B
      #print(logits.shape)
      
      B, T, C = logits.shape
      #print(logits.view(B, -1).shape)
      logits = logits[:, -1, :]# care only about last time dimension
      
      probabilities = F.softmax(logits, dim=-1)# B,C
      #print(probabilities.shape)
      
      next_char = torch.multinomial(probabilities, num_samples=1) # B
      #print(next_char.shape)
      
      
      idx = torch.cat([idx, next_char], dim=1)
      #print(predictions.shape)
      
      #last_idx = next_char

    return idx   
  
model = TransformerEncoderModel()
model = model.to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr = 0.001)


for iter in range(MAX_ITER):
  
  if iter % EVAL_INTERVAL == 0:
    losses = estimate_loss(model)
    print(f"step {iter}: Train loss {losses['train']}, val_loss {losses['val']}")
  
  xb, yb = get_batch('train')
  logits, loss = model(xb, yb)
  opt.zero_grad(set_to_none=True)
  loss.backward()
  opt.step()
print(loss)  

context = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
pred = model.generate(context, 500)
print(decoder(pred[0].tolist()))