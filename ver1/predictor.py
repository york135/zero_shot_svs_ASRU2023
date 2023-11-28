import os, sys, math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import hparams as hp
import transformer.Constants as Constants
from utils import *

class BasePredictLayer(nn.Module):
    def __init__(self, input_size, filter_size, kernel, dropout, dilation=1):
        super(BasePredictLayer, self).__init__()

        self.input_size = input_size
        self.filter_size = filter_size
        self.kernel = kernel
        self.dropout = dropout

        padding = dilation * (self.kernel - 1) // 2

        self.layer_stack = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            dilation=dilation,
                            padding=padding,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            dilation=dilation,
                            padding=padding,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

    def forward(self, spec_input):
        out = self.layer_stack(spec_input)
        return out


class NoteParameter_predictor(nn.Module):
    def __init__(self, feature_dim=hp.encoder_dim, d_model=hp.encoder_dim):

        super(NoteParameter_predictor, self).__init__()

        self.input_size = feature_dim
        self.filter_size = d_model
        self.kernel = 5
        self.dropout = 0

        self.dilation = 1
        self.padding = self.dilation * (self.kernel - 1) // 2

        self.feature_expansion = Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            dilation=self.dilation,
                            padding=self.padding,
                        )
        self.layer_stack1 = BasePredictLayer(self.filter_size, self.filter_size, self.kernel, self.dropout)
        self.layer_stack2 = BasePredictLayer(self.filter_size, self.filter_size, self.kernel, self.dropout)
        self.output_linear = nn.Sequential(
            nn.Linear(d_model, 1),
        )

    def forward(self, note_pitch_cnn_enc):

        note_pitch_cnn_enc = self.feature_expansion(note_pitch_cnn_enc)
        prediction = self.layer_stack1(note_pitch_cnn_enc)
        prediction = self.layer_stack2(prediction)
        # output: [period (sec), phase (-pi~pi)]
        prediction = self.output_linear(prediction)

        return prediction


class Energy_predictor(nn.Module):
    def __init__(self,
                 len_max_seq=hp.vocab_size,
                 n_layers=hp.encoder_n_layer,
                 n_head=hp.encoder_head,
                 d_k=hp.encoder_dim // hp.encoder_head,
                 d_v=hp.encoder_dim // hp.encoder_head,
                 d_model=hp.encoder_dim,
                 d_inner=hp.encoder_conv1d_filter_size,
                 dropout=hp.dropout):

        super(Energy_predictor, self).__init__()

        self.input_size = d_model * 2 + 10
        self.filter_size = d_model
        self.kernel = 3
        self.padding = (self.kernel - 1) // 2
        self.dropout = 0.1

        self.layer_stack = nn.Sequential(
            nn.Conv1d(self.input_size, self.filter_size, kernel_size=self.kernel, padding=self.padding),
            nn.GroupNorm(16, self.filter_size),
            nn.Tanh(),
            nn.Dropout2d(0.1),
            nn.Conv1d(self.filter_size, self.filter_size, kernel_size=self.kernel, padding=self.padding),
            nn.GroupNorm(16, self.filter_size),
            nn.Tanh(),
            nn.Dropout2d(0.1),
            nn.Conv1d(self.filter_size, self.filter_size, kernel_size=self.kernel, padding=self.padding),
        )

        self.ref_emb_to_energy_emb = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

        self.output_linear = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, source_linguistic_encoding, pitch_encoding, ref_encoding_sap):

        ref_energy_emb = self.ref_emb_to_energy_emb(ref_encoding_sap.squeeze(1)).unsqueeze(1)

        model_input = torch.cat((source_linguistic_encoding, pitch_encoding, torch.repeat_interleave(ref_energy_emb, source_linguistic_encoding.shape[1], dim=1)), dim=2)
        model_input = model_input.transpose(1, 2)
        prediction = self.layer_stack(model_input)
        prediction = prediction.transpose(1, 2)
        prediction = self.output_linear(prediction)

        return prediction



class Phoneme_dur_predictor(nn.Module):
    def __init__(self,
                 len_max_seq=hp.vocab_size,
                 d_word_vec=hp.encoder_dim,
                 n_layers=hp.encoder_n_layer,
                 n_head=hp.encoder_head,
                 d_k=hp.encoder_dim // hp.encoder_head,
                 d_v=hp.encoder_dim // hp.encoder_head,
                 d_model=hp.encoder_dim,
                 d_inner=hp.encoder_conv1d_filter_size,
                 dropout=hp.dropout):

        super(Phoneme_dur_predictor, self).__init__()

        n_position = len_max_seq + 1

        self.phoneme_order_embedding = nn.Embedding(20,
                                         10,
                                         padding_idx=Constants.PAD)

        self.input_size = d_model * 3 + 28
        self.filter_size = d_model
        self.kernel = 3
        self.dropout = 0.1

        self.dnn = nn.Sequential(
            nn.Linear(self.input_size, self.filter_size),
            nn.ReLU(),
            nn.Linear(self.filter_size, self.filter_size),
        )

        self.cnn = nn.Sequential(
            nn.Conv1d(self.filter_size, self.filter_size, kernel_size=self.kernel, padding=(self.kernel - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(self.filter_size, self.filter_size, kernel_size=self.kernel, padding=(self.kernel - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(self.filter_size, self.filter_size, kernel_size=self.kernel, padding=(self.kernel - 1) // 2),
        )

        self.output_linear = nn.Sequential(
            nn.Linear(self.filter_size, self.filter_size),
            nn.ReLU(),
            nn.Linear(self.filter_size, 6),
        )

    def forward(self, input_feature, phoneme_embedding, ref_phoneme_emb, phoneme_order, duration_feature):
        phoneme_order = self.phoneme_order_embedding(phoneme_order)

        enc_output = torch.cat((input_feature, phoneme_embedding, phoneme_order, duration_feature
            , torch.repeat_interleave(ref_phoneme_emb, input_feature.shape[1], dim=1)), dim=2)

        enc_output = self.dnn(enc_output).contiguous().transpose(1, 2)
        enc_output = self.cnn(enc_output).transpose(1, 2)
        enc_output = self.output_linear(enc_output)
        return enc_output

class F0_predictor(nn.Module):
    def __init__(self,
                 feature_dim=hp.encoder_dim,
                 n_pitch=hp.pitch_size,
                 len_max_seq=hp.max_seq_len,
                 n_layers=hp.pitch_encoder_n_layer,
                 n_head=hp.encoder_head,
                 d_k=hp.encoder_dim // hp.encoder_head,
                 d_v=hp.encoder_dim // hp.encoder_head,
                 d_model=hp.encoder_dim,
                 d_inner=hp.encoder_conv1d_filter_size,
                 dropout=hp.dropout):
        super(F0_predictor, self).__init__()

        n_position = len_max_seq + 1

        self.input_size = 4 + feature_dim # 16: ref encoder 4: align distance
        self.filter_size = d_model
        self.kernel = 5
        self.padding = (self.kernel - 1) // 2
        self.dropout = 0.2

        self.layer_stack1 = nn.Sequential(
            Conv(self.input_size, self.filter_size, self.kernel, stride=1, padding=self.padding, dilation=1),
            nn.ReLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(self.dropout),
            Conv(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding*2, dilation=2),
            nn.ReLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(self.dropout),
            Conv(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding*4, dilation=4),
            nn.ReLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(self.dropout),
            Conv(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding*8, dilation=8),
        )

        self.layer_stack2 = nn.Sequential(
            Conv(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding, dilation=1),
            nn.ReLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(self.dropout),
            Conv(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding*2, dilation=2),
            nn.ReLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(self.dropout),
            Conv(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding*4, dilation=4),
            nn.ReLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(self.dropout),
            Conv(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding*8, dilation=8),
        )

        self.output_linear = nn.Linear(d_model, 1)
        self.amp_linear = nn.Linear(d_model, 1)


    def forward(self, note_pitch_diff_feature_expand, alignment_distance, f0_parameter_expanded, vib_phase_shift):
        # From 3Hz to 10Hz (divided by 200 to meet the 200 frames/sec hop length)
        vibrato_frequency = (torch.sigmoid(f0_parameter_expanded[:,:,0]) * 7.0 + 3.0) / 50.0
        vibrato_phase = (alignment_distance[:,:,0] * vibrato_frequency + vib_phase_shift) * math.pi * 2.0

        enc_output = torch.cat((note_pitch_diff_feature_expand, alignment_distance), dim=2)

        enc_output = F.relu(self.layer_stack1(enc_output))
        enc_output = F.relu(self.layer_stack2(enc_output))

        f0_residual_residual = self.output_linear(enc_output)

        amp = torch.abs(self.amp_linear(enc_output).squeeze(2)) + 0.01
        vibrato_value = torch.sin(vibrato_phase) * amp
        f0_output = f0_residual_residual + vibrato_value.unsqueeze(2)

        return f0_output, enc_output, f0_residual_residual, amp, vibrato_value

class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
