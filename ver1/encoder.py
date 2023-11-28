import os,re
import numpy as np
import sys
import argparse
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import utils
from utils import process_text, pad_1D, pad_2D
import hparams as hp

import transformer.Constants as Constants
from transformer.Layers import FFTBlock, PreNet, PostNet, Linear

class EncoderLayer(nn.Module):
    def __init__(self, d_in, d_hid, kernel, padding, dropout, dilation=1, use_ln=True):
        super().__init__()
        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=kernel, padding=padding*dilation, dilation=dilation)
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=1, padding=0, dilation=dilation)

        self.use_ln = use_ln

        if self.use_ln:
            self.layer_norm = nn.LayerNorm(d_in)

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        residual = x
        output = self.w_2(F.leaky_relu(self.w_1(F.leaky_relu(x, negative_slope=0.2)), negative_slope=0.2))
        output = self.dropout(output) + residual
        if self.use_ln:
            output = self.layer_norm(output.transpose(1, 2)).transpose(1, 2)
        return output

class FloatPitchEncoder(nn.Module):
    ''' Encoder '''

    def __init__(self,
                 n_pitch=hp.pitch_size,#
                 len_max_seq=hp.max_seq_len,
                 n_layers=hp.pitch_encoder_n_layer,#
                 n_head=hp.encoder_head,
                 d_k=hp.encoder_dim // hp.encoder_head,
                 d_v=hp.encoder_dim // hp.encoder_head,
                 d_model=hp.encoder_dim,
                 d_inner=hp.encoder_conv1d_filter_size,
                 dropout=hp.dropout):

        super(FloatPitchEncoder, self).__init__()

        n_position = len_max_seq + 1
        
        self.input_size = d_model
        self.filter_size = d_model
        self.pre_output_size = d_model
        self.kernel = 9
        self.dropout = 0.1

        self.padding = (self.kernel - 1) // 2

        self.pre_layer1 = EncoderLayer(self.input_size, self.filter_size, self.kernel, self.padding, self.dropout, use_ln=True)
        self.pre_layer2 = EncoderLayer(self.filter_size, self.filter_size, self.kernel, self.padding, self.dropout, use_ln=True)
        self.pre_layer3 = EncoderLayer(self.filter_size, self.filter_size, self.kernel, self.padding, self.dropout, use_ln=True)
        self.pre_layer4 = EncoderLayer(self.filter_size, self.filter_size, self.kernel, self.padding, self.dropout, use_ln=True)
        self.pre_layer5 = EncoderLayer(self.filter_size, self.pre_output_size, self.kernel, self.padding, self.dropout, use_ln=True)

        self.layer_stack = nn.ModuleList([FFTBlock(
            d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(1)])

    def forward(self, input_feature, src_pos):
        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_pos, seq_q=src_pos)
        non_pad_mask = get_non_pad_mask(src_pos)

        # -- Forward
        enc_output = input_feature.contiguous().transpose(1, 2)

        enc_output = self.pre_layer1(enc_output)
        enc_output = self.pre_layer2(enc_output)
        enc_output = self.pre_layer3(enc_output)
        enc_output = self.pre_layer4(enc_output)
        enc_output = self.pre_layer5(enc_output).contiguous().transpose(1, 2)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output, non_pad_mask


class NoteEncoder(nn.Module):
    def __init__(self,
                 n_pitch=hp.pitch_size,
                 d_model=hp.encoder_dim):
        super(NoteEncoder, self).__init__()

        # for note pitch
        self.src_pitch_emb = nn.Embedding(n_pitch,
                                         d_model,
                                         padding_idx=Constants.PAD)

        self.input_size = d_model * 2
        self.filter_size = d_model
        self.kernel = 5

        self.ref_emb_to_pitch_emb = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        
        self.layer_stack1 = nn.Sequential(
            nn.Conv1d(self.filter_size + 10, self.filter_size, kernel_size=self.kernel, padding=(self.kernel - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(self.filter_size, self.filter_size, kernel_size=self.kernel, padding=(self.kernel - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(self.filter_size, self.filter_size, kernel_size=self.kernel, padding=(self.kernel - 1) // 2)
        )


    def forward(self, note_pitch, ref_encoding_sap):

        note_pitch_embed = self.src_pitch_emb(note_pitch)

        ref_pitch_emb = self.ref_emb_to_pitch_emb(ref_encoding_sap.squeeze(1)).unsqueeze(1)

        note_pitch_embed = torch.cat((note_pitch_embed, torch.repeat_interleave(ref_pitch_emb, note_pitch_embed.shape[1], dim=1)), dim=2)

        enc_output = note_pitch_embed.contiguous().transpose(1, 2)
        enc_output = self.layer_stack1(enc_output)
        enc_output = enc_output.contiguous().transpose(1, 2)
        return note_pitch_embed, enc_output