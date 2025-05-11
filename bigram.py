import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# Set Parameters
batch_size = 64
block_size = 16
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

# Set the largest substring allowed to be tokenized
max_char_seq = 1
# Percent of data used for training
training_size = 0.8

torch.manual_seed(1234)

# Load the file that we will be using
with open("dexter_transcript/season_1/season_1_transcript_filtered.txt", 'r') as f:
    file_content = f.read()

# Create a tokenizer dictionary with the maximum token length
# This helps to reduce the number of tokens needed for any arbitrary string
str_to_int = {}
token_count = 0
for length in range(1, max_char_seq + 1):
    sub_strings = []
    for i in range(len(file_content) - length + 1):
        sub_strings.append(file_content[i:i+length])
    for i, substring in enumerate(list(set(sub_strings)), token_count):
        str_to_int[substring] = i
    token_count = len(str_to_int)
    print(f"Length : {length} Token Count : {token_count}")

int_to_str = dict(zip(str_to_int.values(), str_to_int.keys()))
    
# Define the char -> int encoding function
def encode(s):
    encoding = []
    max_token_length = max_char_seq if len(s) >= max_char_seq else len(s)
    idx = 0
    while idx < len(s):
        if s[idx:max_token_length+idx] in str_to_int:
            encoding.append(str_to_int[s[idx:max_token_length + idx]])
            idx += max_token_length
        else:
            max_token_length -= 1
    return encoding

# Define the int -> char decoding function
def decode(int_list):
    return ''.join([int_to_str[i] for i in int_list])

# Encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(file_content), dtype=torch.long)
n = int(training_size * len(data))
train_data = data[:n]
val_data = data[n:]

# Generates a small batch of data of inputs and targets (x,y)
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            pred, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLanguageModel(nn.Module):
    def __init__(self, token_count):
        super().__init__()
        self.token_embedding_table = nn.Embedding(token_count, token_count)

    def forward(self, idx, targets=None):
        pred = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            batch, block, tokens = pred.shape
            pred = pred.view(batch*block, tokens)
            targets = targets.view(batch*block)
            loss = F.cross_entropy(pred, targets)

        return pred, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (batch, block) array of indices in current context
        for _ in range(max_new_tokens):
            pred, loss = self(idx)
            # using last block get probabilities
            pred = pred[:, -1, :]
            prob = F.softmax(pred, dim=1)
            # sample from the distribution
            idx_next = torch.multinomial(prob, num_samples=1)
            # add sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

print("\n----------\n")
model = BigramLanguageModel(token_count)
m = model.to(device)

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if not iter % eval_interval:
        losses = estimate_loss()
        print(f"Step {iter} : training loss {losses['train']:.4f}, validation loss {losses['val']:.4f}")
    
    # sample data
    xb, yb = get_batch('train')

    # evaluate loss
    pred, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))