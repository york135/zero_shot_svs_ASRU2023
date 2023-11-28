import torch
import torch.nn as nn
import numpy as np
import hparams as hp

import sys, os

import transformer.Constants as Constants
from transformer.Layers import FFTBlock, FFTBlockAdaLN, PreNet, PostNet, Linear

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

class LyricsEncoder(nn.Module):
    ''' Encoder '''

    def __init__(self,
                 n_src_vocab=hp.vocab_size,
                 len_max_seq=hp.vocab_size,
                 d_word_vec=hp.encoder_dim,
                 n_layers=hp.encoder_n_layer,
                 n_head=hp.encoder_head,
                 d_k=hp.encoder_dim // hp.encoder_head,
                 d_v=hp.encoder_dim // hp.encoder_head,
                 d_model=hp.encoder_dim,
                 d_inner=hp.encoder_conv1d_filter_size,
                 dropout=hp.dropout):

        super(LyricsEncoder, self).__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(n_src_vocab,
                                         d_word_vec,
                                         padding_idx=Constants.PAD)

        # Tone: at most 10
        self.src_tone_emb = nn.Embedding(10,
                                         d_word_vec,
                                         padding_idx=Constants.PAD)


        self.kernel = 3
        self.pre_cnn_1220 = nn.Sequential(
            nn.Conv1d(d_word_vec + 18, d_word_vec, kernel_size=self.kernel, padding=(self.kernel - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(d_word_vec, d_word_vec, kernel_size=self.kernel, padding=(self.kernel - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(d_word_vec, d_word_vec, kernel_size=self.kernel, padding=(self.kernel - 1) // 2)
        )

        self.ref_emb_to_phoneme_emb = nn.Sequential(
            nn.Linear(d_word_vec, d_word_vec),
            nn.ReLU(),
            nn.Linear(d_word_vec, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

        self.post_cnn = nn.Sequential(
            nn.Conv1d(d_word_vec, d_word_vec, kernel_size=self.kernel, padding=(self.kernel - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(d_word_vec, d_word_vec * 2, kernel_size=self.kernel, padding=(self.kernel - 1) // 2)
        )

    def forward(self, phoneme_seq, ref_encoding_sap, phoneme_tone, duration_feature, src_pos, return_attns=False):
        # -- Forward
        phoneme_embedding = self.src_word_emb(phoneme_seq) + self.src_tone_emb(phoneme_tone)
        enc_output = phoneme_embedding

        ref_phoneme_emb = self.ref_emb_to_phoneme_emb(ref_encoding_sap.squeeze(1)).unsqueeze(1)

        enc_output = torch.cat((enc_output, duration_feature, torch.repeat_interleave(ref_phoneme_emb, enc_output.shape[1], dim=1)), dim=2).contiguous().transpose(1, 2)
        enc_output = self.pre_cnn_1220(enc_output)

        enc_output = self.post_cnn(enc_output).transpose(1, 2)
        return enc_output, phoneme_embedding, ref_phoneme_emb
