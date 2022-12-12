import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout, num_class, device):
        super(RNN, self).__init__()

        self.device = device        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(embed_dim, self.hidden_dim, self.n_layers, batch_first=True)

        self.linear_layer = nn.Linear(self.hidden_dim, num_class)

    def forward(self, sentence):
        x = self.embed(sentence)

        init_hidden = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(self.device)

        output, _ = self.gru(x, init_hidden)

        t_hidden = output[:, -1, :]

        self.dropout(t_hidden)

        logits = self.linear_layer(t_hidden)

        return logits