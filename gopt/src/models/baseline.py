# -*- coding: utf-8 -*-
# @Time    : 1/8/22 2:27 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : baseline.py

import math
import warnings
import torch
import torch.nn as nn
import numpy as np

class BaselineLSTM(nn.Module):
    def __init__(self, embed_dim, depth, input_dim=84, extra_dim=0):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.extra_dim = extra_dim
        head_dim = embed_dim + extra_dim
        self.lstm = torch.nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, num_layers=depth, batch_first=True)

        # for phone classification
        self.in_proj = nn.Linear(self.input_dim, embed_dim)
        self.mlp_head_phn = nn.Sequential(nn.LayerNorm(head_dim), nn.Linear(head_dim, 1))

        # for word classification
        self.mlp_head_word1 = nn.Sequential(nn.LayerNorm(head_dim), nn.Linear(head_dim, 1))
        self.mlp_head_word2 = nn.Sequential(nn.LayerNorm(head_dim), nn.Linear(head_dim, 1))
        self.mlp_head_word3 = nn.Sequential(nn.LayerNorm(head_dim), nn.Linear(head_dim, 1))

        # phone projection
        self.phn_proj = nn.Linear(40, embed_dim)

        # utterance level
        self.mlp_head_utt1 = nn.Sequential(nn.LayerNorm(head_dim), nn.Linear(head_dim, 1))
        self.mlp_head_utt2 = nn.Sequential(nn.LayerNorm(head_dim), nn.Linear(head_dim, 1))
        self.mlp_head_utt3 = nn.Sequential(nn.LayerNorm(head_dim), nn.Linear(head_dim, 1))
        self.mlp_head_utt4 = nn.Sequential(nn.LayerNorm(head_dim), nn.Linear(head_dim, 1))
        self.mlp_head_utt5 = nn.Sequential(nn.LayerNorm(head_dim), nn.Linear(head_dim, 1))

    # get the output of the last valid token
    def get_last_valid(self, input, mask):
        output = []
        B = input.shape[0]
        seq_len = input.shape[1]
        for i in range(B):
            for j in range(seq_len):
                if mask[i, j] == 0:
                    output.append(input[i, j-1])
                    break
                if j == seq_len - 1:
                    print('append')
                    output.append(input[i, j])
        output = torch.stack(output, dim=0)
        return output.unsqueeze(1)

    # x shape in [batch_size, sequence_len, feat_dim]
    # phn in [batch_size, seq_len]
    def forward(self, x, phn, extra_feat=None):

        # batch size
        B = x.shape[0]
        seq_len = x.shape[1]
        valid_tok_mask = (phn>=0)

        # phn_one_hot in shape [batch_size, seq_len, feat_dim]
        phn_one_hot =  torch.nn.functional.one_hot(phn.long()+1, num_classes=40).float()
        # phn_embed in shape [batch_size, seq_len, embed_dim]
        phn_embed = self.phn_proj(phn_one_hot)

        if self.embed_dim != self.input_dim:
            x = self.in_proj(x)

        x = x + phn_embed

        x = self.lstm(x)[0]

        def fuse_hidden(hidden, extra):
            if self.extra_dim == 0:
                return hidden
            if extra is None:
                raise ValueError('extra_feat is required when extra_dim > 0')
            if hidden.dim() == 2:
                return torch.cat((hidden, extra), dim=1)
            extra_expand = extra.unsqueeze(1).repeat(1, hidden.shape[1], 1)
            return torch.cat((hidden, extra_expand), dim=2)

        token_hidden = fuse_hidden(x, extra_feat)

        p = self.mlp_head_phn(token_hidden).reshape(B, seq_len, 1)

        w1 = self.mlp_head_word1(token_hidden).reshape(B, seq_len, 1)
        w2 = self.mlp_head_word2(token_hidden).reshape(B, seq_len, 1)
        w3 = self.mlp_head_word3(token_hidden).reshape(B, seq_len, 1)

        u1 = self.get_last_valid(self.mlp_head_utt1(token_hidden).reshape(B, seq_len), valid_tok_mask)
        u2 = self.get_last_valid(self.mlp_head_utt2(token_hidden).reshape(B, seq_len), valid_tok_mask)
        u3 = self.get_last_valid(self.mlp_head_utt3(token_hidden).reshape(B, seq_len), valid_tok_mask)
        u4 = self.get_last_valid(self.mlp_head_utt4(token_hidden).reshape(B, seq_len), valid_tok_mask)
        u5 = self.get_last_valid(self.mlp_head_utt5(token_hidden).reshape(B, seq_len), valid_tok_mask)

        return u1, u2, u3, u4, u5, p, w1, w2, w3