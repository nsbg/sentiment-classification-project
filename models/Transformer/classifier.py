import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from .encoder import Encoder, EncoderLayer
from .sublayers import *

attn = MultiHeadAttention(8, 152)
ff = PositionwiseFeedForward(152, 1024, 0.5)
pe = PositionalEncoding(152, 0.5)

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, num_class):
        super(Transformer, self).__init__()

        self.encoder = Encoder(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff)), n_layer)
        self.src_embed = nn.Sequential(Embeddings(d_model, vocab_size), deepcopy(pe))
        self.linear = nn.Linear(d_model, num_class)

    def forward(self, x):
        x = self.src_embed(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        x = self.linear(x)
        
        logits = F.softmax(x, dim=-1)

        return logits