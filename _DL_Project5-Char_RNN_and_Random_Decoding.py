#!/usr/bin/env python
# coding: utf-8

# In[92]:


from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# ### Get the data and process
# - This is the Mysterious island found in Project Gutenberg.

# In[93]:


## Reading and processing text
with open('1268-0.txt', 'r', encoding="utf8") as fp:
    text=fp.read()

# Get the index of 'THE MYSTERIOUS ISLAND' or 'The Mysterious Island'
start_indx = text.find('THE MYSTERIOUS ISLAND') if text.find('THE MYSTERIOUS ISLAND') != -1 else text.find('The Mysterious Island')
# Get the index of 'End of the Project Gutenberg'
end_indx = text.find('End of the Project Gutenberg') 


# Set text to the text between start and end idx.
text = text[start_indx:end_indx]
# Get the unique set of characters.
char_set = set(text)
print('Total Length:', len(text))
print('Unique Characters:', len(char_set))
assert(len(text) == 1130711)
assert(len(char_set) == 85)


# ### Tokenze and get other helpers
# - We do this manually since everything is character based.

# In[94]:


# The universe of words.
chars_sorted = sorted(char_set)

# Effectively, these maps are the tokenizer.
# Map each char to a unique int. This is a dict.
char2int = {key:item for item,key in enumerate(list(chars_sorted))}
# Do the revverse of the above, this should be a np array.
int2char = np.array(chars_sorted)

# Tokenize the entire corpus. This should be an np array of np.int32 type.
text_encoded = np.array([char2int[char] for char in text], dtype=np.int32)

print('Text encoded shape: ', text_encoded.shape)

print(text[:15], '     == Encoding ==> ', text_encoded[:15])
print(text_encoded[15:21], ' == Reverse  ==> ', ''.join(int2char[text_encoded[15:21]]))


# #### Examples

# In[95]:


print('Text encoded shape: ', text_encoded.shape)
print(text[:15], '     == Encoding ==> ', text_encoded[:15])
print(text_encoded[15:21], ' == Reverse  ==> ', ''.join(int2char[text_encoded[15:21]]))


# In[96]:


assert(
    np.array_equal(
    text_encoded[:15],
        [48, 36, 33, 1, 41, 53, 47, 48, 33, 46, 37, 43, 49, 47,  1]
    )
)


# ### Process the data and get the data loader

# In[97]:


seq_length = 40
chunk_size = seq_length + 1

# Break up the data into chunks of size 41. This should be a list of lists.
# Use text_encoded. This will be used to get (x, y) pairs.
text_chunks = []

for i in range(0,len(text_encoded)-seq_length,1):
    text_chunks.append(text_encoded[i:i+chunk_size].tolist())
    


# In[98]:


class TextDataset(Dataset):
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks

    def __len__(self):
        return len(self.text_chunks)
    
    def __getitem__(self, idx):
        # Get the text chunk at index idx.
        text_chunk = self.text_chunks[idx]
        # Return (x, y) where x has length 40 and y has length 40.
        # y should be x shifted by 1 time.
        return (text_chunk[:40], text_chunk[1:41])
    
seq_dataset = TextDataset(torch.tensor(text_chunks))


# In[99]:


for i, (seq, target) in enumerate(seq_dataset):
    # 40 characters for source and target 
    print(seq.shape, target.shape)
    print('Input (x):', repr(''.join(int2char[seq])))
    print('Target (y):', repr(''.join(int2char[target])))
    print()
    if i == 1:
        break 


# In[100]:


device = torch.device("cpu")


# In[101]:


batch_size = 64
torch.manual_seed(1)
seq_dl = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# ### Write the models

# In[82]:


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size):
        super().__init__()
        # Set to an embedding layer of vocab_size by embed_dim.
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_hidden_size = rnn_hidden_size
        # Set to an LSTM with x having embed_dim and h dimension rnn_hidden_size.
        # batch_first shoould be true.
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        
        # Make a linear layer from rnn_hidden_size to vocab_size.
        # This will be used to get the yt for each xt.
        self.fc = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, text, hidden=None, cell=None):
        # Get the embeddings for text.
        out = self.embedding(text)
        
        # Pass out, hidden and cell through the rnn.
        # If hidden is None, don't specify it and just use out.
        if hidden is not None:
            out, (hidden, cell) = self.rnn(out,(hidden,cell))
        else:
            out, (hidden, cell) = self.rnn(out)
        
        # Pass out through fc.
        out = self.fc(out)
        
        return out, (hidden, cell)

    def init_hidden(self, batch_size):
        # Initialize to zeros of 1 by ??? appropriate dimensions.
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden.to(device), cell.to(device)


# In[ ]:





# ### Do this right way - across all data all at once!

# In[102]:


vocab_size = len(int2char)
embed_dim = 256
rnn_hidden_size = 512

torch.manual_seed(1)
model = RNN(vocab_size, embed_dim, rnn_hidden_size) 
model = model.to(device)
model


# In[104]:


get_ipython().run_cell_magic('time', '', "criterion = nn.CrossEntropyLoss()\noptimizer = torch.optim.Adam(model.parameters(), lr=0.005)\n\n# Set to 10000.\nnum_epochs = 1000\n\ntorch.manual_seed(1)\n\n# epochs here will mean batches.\n# If the above takes too long, use 1000. \n\n## 1000 epochs took 10 mins 11sec, and 10000 may take about 2 hours. We use 1000 here...\nfor epoch in range(num_epochs):\n    hidden, cell = model.init_hidden(batch_size)\n    \n    # Get the next batch from seq_dl\n    seq_batch, target_batch = next(iter(seq_dl))\n        \n    seq_batch = seq_batch.to(device)\n    target_batch = target_batch.to(device)\n    \n    optimizer.zero_grad()\n    \n    loss = 0\n\n    # Pass through the model.\n    logits, _ = model(seq_batch,hidden,cell)\n    \n    # Get the loss.\n    loss += criterion(logits.view(-1, logits.size(2)), target_batch.view(-1))\n        \n    # Do back prop.\n    loss.backward()\n    optimizer.step()\n    \n    # Get the value in the tensor loss.\n    loss = loss.item()\n    \n    if epoch % 100 == 0:\n        print(f'Epoch {epoch} loss: {loss:.4f}')")


# In[105]:


from torch.distributions.categorical import Categorical

torch.manual_seed(1)

logits = torch.tensor([[-1.0, 1.0, 3.0]])

# Get the probabilities for these logits.
p = nn.functional.softmax(logits, dim=1)
print('Probabilities:', p)

# Get a Categorical random variable with the above probabilities for each of the classes.
m = Categorical(p)
# Generate 10 things.
samples = m.sample((10,))
 
print(samples.numpy())


# ### Random decoding.
# - The compounds problems: once we make a mistake, we can't undo it.

# In[110]:


def random_sample(
    model,
    starting_str, 
    len_generated_text=500, 
):

    # Encode starting string into a tensor using char2str.
    encoded_input = torch.tensor([char2int[s] for s in starting_str])
    
    encoded_input = encoded_input.view(1, -1)

    generated_str = starting_str

    # Put model in eval mode. This matters if we had dropout o batch / layer norms.
    model.eval()
    
    hidden, cell = model.init_hidden(1)
    
    hidden = hidden.to(device)
    
    cell = cell.to(device)
        
    # Build up the starting hidden and cell states.
    for c in range(len(starting_str)-1):
        # Feed each letter 1 by 1 and then get the final hidden state.
        out = encoded_input[:, c].view(1, 1)
        # Pass out through, note we update hidden and cell and use them again
        _, (hidden, cell) = model(out,hidden,cell)
    
    # Gte the last char; note we did not do go to the last char above.
    last_char = encoded_input[:, -1]
    # Generate chars one at a time, add them to generated_str.
    for i in range(len_generated_text):
        
        # Use hidden and cell from the above.
        # Use last_char, which will be updated over and over.
        lcout = last_char.view(1, 1)
        logits, (hidden, cell) = model(lcout, hidden, cell)
        
        # Get the logits.
        logits = logits.view(-1)
        
        # m is a random variable with probabilities based on the softmax of the logits.
        m = Categorical(logits = logits)
        
        # Generate from m 1 char.
        last_char = m.sample((1,))
        
        # Add the geenrated char to generated_str, but pass it through int2str so that 
        generated_str += str(int2char[last_char])
        
    return generated_str

torch.manual_seed(1)
model.to(device)
print(random_sample(model, starting_str='The island'))

