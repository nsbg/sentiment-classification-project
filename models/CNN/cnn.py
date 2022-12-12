import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class CNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_filters, filter_size,  dropout, num_class):
        super(CNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1d_layers = nn.ModuleList([nn.Conv1d(in_channels=embed_dim, out_channels=n_filters[i], kernel_size=filter_size[i]) for i in range(len(filter_size))])
        self.fc_layer = nn.Linear(np.sum(n_filters), num_class)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sentence):
        x_embed = self.embedding(sentence)
        x_embed = x_embed.permute(0, 2, 1)

        x_conv_list = [F.relu(conv1d(x_embed)) for conv1d in self.conv1d_layers]
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list]
        
        x_fc_layer = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)

        logits = self.fc_layer(self.dropout(x_fc_layer))

        return logits