import os,re
import collections
import numpy as np
import sys

import argparse
import time
import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#FastSpeech
import utils
from utils import process_text, pad_1D, pad_2D
import hparams as hp

import transformer.Constants as Constants
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward
from transformer.Models import LyricsEncoder

from encoder import FloatPitchEncoder, NoteEncoder
from utils import *
from predictor import Conv, F0_predictor, Phoneme_dur_predictor, Energy_predictor, NoteParameter_predictor

from collections import OrderedDict

from audio_encoder import MelFeatureExtractor, MelTargetFeatureExtractor

sys.path.append(os.path.join(os.path.dirname(__file__), '../share/pwg_pretrained'))
from parallel_wavegan.datasets import MelDataset
from parallel_wavegan.datasets import MelSCPDataset
from parallel_wavegan.utils import load_model
from parallel_wavegan.utils import read_hdf5

from typing import Tuple, List, Optional


def create_soft_alignment(x, alignment, phoneme_seq, is_multi_phoneme, note_duration, duration, device):
    batch_size, L, _ = duration.shape
    note_num = note_duration.shape[1]
    phoneme_num = int(x.shape[-1] / 2)

    # alignment shape: (n, frame num, phoneme num)
    # x shape: (n, phoneme num, encoder dim * 2)
    # output shape: (n, frame num, encoder dim)
    output = torch.zeros((batch_size, alignment.shape[-2], phoneme_num), requires_grad=True).to(device)

    for i in range(batch_size):

        count = 0
        phoneme_start = 0
        phoneme_end = 0

        for j in range(note_num):

            # The phoneme id of <sep> is 1
            while phoneme_start < len(phoneme_seq[i]) - 1 and phoneme_seq[i][phoneme_start] <= 1:
                phoneme_start = phoneme_start + 1

            phoneme_end = phoneme_start

            while phoneme_seq[i][phoneme_end] > 1 and phoneme_end < len(phoneme_seq[i]) - 1:
                phoneme_end = phoneme_end + 1

            start = count
            end = count + note_duration[i][j]
            if j > 0:
                start = start - min(note_duration[i][j-1], 10)
            if j < note_num - 1:
                end = end + min(note_duration[i][j+1], 10)

            start_idx = start - count
            end_idx = end - count

            frame_index_in_note = torch.arange(start_idx, end_idx, step=1).to(device)

            # compute gaussian pdf
            for n in range(phoneme_start, phoneme_end):
                mean_value = torch.clip(duration[i][n][0], min=-0.25, max=1.25) * note_duration[i][j]
                stddev = torch.exp(duration[i][n][1]) + 1.0
                amplitude = torch.exp(duration[i][n][2])
                exp_elements = -torch.pow(frame_index_in_note - mean_value, 2) / (2.0 * stddev * stddev)
                exp_elements = torch.clip(exp_elements, max=20)
                coeff_weights = (1.0 / stddev) * torch.exp(exp_elements)

                cur_alignment = coeff_weights * amplitude
                alignment[i, start:end, n] = alignment[i, start:end, n] + cur_alignment
                output[i, start:end, :] = output[i, start:end, :] + cur_alignment.unsqueeze(1) * x[i, n, :phoneme_num]

                if is_multi_phoneme[i][n] == True:
                    mean_value = torch.clip(duration[i][n][3], min=-0.25, max=1.25) * note_duration[i][j]
                    stddev = torch.exp(duration[i][n][4]) + 1.0
                    amplitude = torch.exp(duration[i][n][5])
                    exp_elements = -torch.pow(frame_index_in_note - mean_value, 2) / (2.0 * stddev * stddev)
                    exp_elements = torch.clip(exp_elements, max=20)
                    coeff_weights = (1.0 / stddev) * torch.exp(exp_elements)

                    cur_alignment = coeff_weights * amplitude
                    alignment[i, start:end, n] = alignment[i, start:end, n] + cur_alignment
                    output[i, start:end, :] = output[i, start:end, :] + cur_alignment.unsqueeze(1) * x[i, n, phoneme_num:]

            count = count + note_duration[i][j]
            phoneme_start = phoneme_end
    
    alignment_norm = torch.norm(alignment, p=1, dim=2, keepdim=True)
    output = output / torch.clip(alignment_norm, min=1e-10)
    alignment = F.normalize(alignment, p=1, dim=2)
    return output, alignment


class PWGVocoder(nn.Module):
    def __init__(self, device='cpu', normalize_path='./pwg_pretrained/stats.h5', vocoder_path='./pwg_pretrained/checkpoint-400000steps.pkl'):
        super(PWGVocoder, self).__init__()
        
        self.device = device

        self.vocoder = load_model(vocoder_path)
        self.vocoder.remove_weight_norm()
        self.vocoder = self.vocoder.eval().to(self.device)


        stat_data = np.load(normalize_path)

        self.normalize_mean = torch.tensor(stat_data[0]).to(self.device)
        self.normalize_scale = torch.tensor(stat_data[1]).to(self.device)

        # Freeze vocoder weight
        for p in self.vocoder.parameters():
            p.requires_grad = False

        self.max_vocoder_segment_length = 400

    def forward(self, spec_output, output_all=False):
        # Go through the vocoder to generate waveform
        spec_output_norm = (spec_output - self.normalize_mean) / self.normalize_scale

        # Pick at most "self.max_vocoder_segment_length" frames, in order to avoid CUDA OOM.
        # x is the random noise for vocoder
        if spec_output_norm.shape[1] > self.max_vocoder_segment_length and output_all == False:
            start_frame = int(torch.rand(1) * (spec_output_norm.shape[1] - self.max_vocoder_segment_length))
            end_frame = start_frame + self.max_vocoder_segment_length
            spec_for_vocoder = torch.nn.ReplicationPad1d(2)(spec_output_norm[:,start_frame:end_frame,:].transpose(1, 2))
            x = torch.randn(spec_output_norm.shape[0], 1, self.max_vocoder_segment_length * self.vocoder.upsample_factor).to(self.device)
        else:
            start_frame = 0
            spec_for_vocoder = torch.nn.ReplicationPad1d(2)(spec_output_norm.transpose(1, 2))
            x = torch.randn(spec_output_norm.shape[0], 1, spec_output_norm.shape[1] * self.vocoder.upsample_factor).to(self.device)
        
        waveform_output = self.vocoder(x, spec_for_vocoder).squeeze(1)

        return waveform_output, start_frame


class Smoother(nn.Module):
    """Convolutional Transformer Encoder Layer"""

    def __init__(self, d_model: int, nhead: int, d_hid: int, dropout=0.1):
        super(Smoother, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.conv1 = nn.Conv1d(d_model, d_hid, 9, padding=4)
        self.conv2 = nn.Conv1d(d_hid, d_model, 1, padding=0)

        self.norm1 = nn.GroupNorm(16, hp.encoder_dim)
        self.norm2 = nn.GroupNorm(16, hp.encoder_dim)
        self.dropout1 = nn.Dropout2d(dropout)
        self.dropout2 = nn.Dropout2d(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal = False,
    ) -> torch.Tensor:
        # multi-head self attention
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]

        src2 = src2.transpose(0, 1).transpose(1, 2)
        src = src.transpose(0, 1).transpose(1, 2)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # conv1d
        src2 = self.conv2(F.relu(self.conv1(src)))

        # add & norm
        src = src + self.dropout2(src2)
        src = self.norm2(src).transpose(1, 2).transpose(0, 1)
        return src

class DecoderPostNet(nn.Module):
    def __init__(self, device):
        super(DecoderPostNet, self).__init__()
        self.device = device

        self.dec_net1 = nn.Sequential(
            nn.Conv1d(hp.encoder_dim * 3 + 1, hp.encoder_dim, kernel_size=5, padding=2),
            nn.GroupNorm(16, hp.encoder_dim),
            nn.Tanh(),
            nn.Dropout2d(0.1),
            nn.Conv1d(hp.encoder_dim, hp.encoder_dim, kernel_size=5, padding=2),
            nn.GroupNorm(16, hp.encoder_dim),
            nn.Tanh(),
            nn.Dropout2d(0.1),
            nn.Conv1d(hp.encoder_dim, hp.encoder_dim, kernel_size=5, padding=2),
        )

        self.dec_net2 = nn.Sequential(
            nn.Conv1d(hp.encoder_dim, hp.encoder_dim, kernel_size=5, padding=2),
            nn.GroupNorm(16, hp.encoder_dim),
            nn.Tanh(),
            nn.Dropout2d(0.1),
            nn.Conv1d(hp.encoder_dim, hp.encoder_dim, kernel_size=5, padding=2),
            nn.GroupNorm(16, hp.encoder_dim),
            nn.Tanh(),
            nn.Dropout2d(0.1),
            nn.Conv1d(hp.encoder_dim, hp.encoder_dim, kernel_size=5, padding=2),
        )

        self.smoothers = nn.TransformerEncoder(Smoother(hp.encoder_dim, 2, 256, dropout=0.1), num_layers=2)

        self.spec_linear = nn.Conv1d(hp.encoder_dim, hp.num_spec, kernel_size=5, padding=2)

        self.dec_post_net = nn.Sequential(
            nn.Conv1d(hp.num_spec, hp.encoder_dim, kernel_size=5, padding=2),
            nn.GroupNorm(16, hp.encoder_dim),
            nn.Tanh(),
            nn.Dropout2d(0.1),
            nn.Conv1d(hp.encoder_dim, hp.encoder_dim, kernel_size=5, padding=2),
            nn.GroupNorm(16, hp.encoder_dim),
            nn.Tanh(),
            nn.Dropout2d(0.1),
            nn.Conv1d(hp.encoder_dim, hp.encoder_dim, kernel_size=5, padding=2),
            nn.GroupNorm(16, hp.encoder_dim),
            nn.Tanh(),
            nn.Dropout2d(0.1),
            nn.Conv1d(hp.encoder_dim, hp.num_spec, kernel_size=5, padding=2),
        )

    def forward(self, decoder_input, pitch_encoding, ref_encoding_sap, energy_feature):
        decoder_input = torch.cat((decoder_input, pitch_encoding, torch.repeat_interleave(ref_encoding_sap, decoder_input.shape[1], dim=1), energy_feature), dim=2)

        decoder_input = decoder_input.transpose(1, 2)
        decoder_output = self.dec_net1(decoder_input)
        decoder_output = decoder_output + self.dec_net2(decoder_output)
        decoder_output = decoder_output.transpose(1, 2).transpose(0, 1)
        decoder_output = self.smoothers(decoder_output).transpose(0, 1).transpose(1, 2)

        decoder_output = self.spec_linear(decoder_output)

        decoder_output_after = decoder_output + self.dec_post_net(decoder_output)

        decoder_output = decoder_output.transpose(1, 2)
        decoder_output_after = decoder_output_after.transpose(1, 2)

        return decoder_output, decoder_output_after


class Score_dur_to_note_dur(nn.Module):
    ''' Encoder '''

    def __init__(self, device):
        super(Score_dur_to_note_dur, self).__init__()

        self.n_src_vocab = hp.vocab_size
        self.d_word_vec = hp.encoder_dim
        self.device = device

        # Phoneme-level embedding -> Note-level
        self.src_word_emb = nn.Embedding(self.n_src_vocab,
                                         self.d_word_vec,
                                         padding_idx=Constants.PAD)

        self.src_pos_emb = nn.Embedding(20,
                                         10,
                                         padding_idx=Constants.PAD)

        self.kernel = 3

        self.phoneme_mix = nn.Sequential(
            nn.Conv1d(self.d_word_vec + 10, self.d_word_vec, kernel_size=self.kernel, padding=(self.kernel - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(self.d_word_vec, self.d_word_vec, kernel_size=self.kernel, padding=(self.kernel - 1) // 2)
        )

        # Note-level -> Time-lag model (Note dur difference)
        self.time_lag_rnn = nn.LSTM(
            input_size=self.d_word_vec + 2,
            hidden_size=self.d_word_vec,
            batch_first=True,
            num_layers=2,
            dropout=0.1,
            bidirectional=True
        )

        self.time_lag_cnn = nn.Sequential(
            nn.Conv1d(self.d_word_vec * 2, self.d_word_vec, kernel_size=self.kernel, padding=(self.kernel - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(self.d_word_vec, 1, kernel_size=self.kernel, padding=(self.kernel - 1) // 2)
        )

    def forward(self, score_note_dur, phoneme_seq, phoneme_order):
        phoneme_embedding = self.src_word_emb(phoneme_seq)
        phoneme_pos_embedding = self.src_pos_emb(phoneme_order)
        phoneme_embedding = torch.cat((phoneme_embedding, phoneme_pos_embedding), dim=2).transpose(1, 2)
        phoneme_embedding = self.phoneme_mix(phoneme_embedding).transpose(1, 2)

        batch_size, phoneme_num, _ = phoneme_embedding.shape
        note_num = score_note_dur.shape[1]

        note_level_aggregation = torch.zeros((score_note_dur.shape[0], score_note_dur.shape[1], self.d_word_vec), device=self.device)

        for i in range(batch_size):
            count = 0
            phoneme_start = 0
            phoneme_end = 0

            for j in range(note_num):
                # The phoneme id of <sep> is 1, padding is 0
                while phoneme_start < len(phoneme_seq[i]) - 1 and phoneme_seq[i][phoneme_start] <= 1:
                    phoneme_start = phoneme_start + 1

                phoneme_end = phoneme_start

                while phoneme_seq[i][phoneme_end] > 1 and phoneme_end < len(phoneme_seq[i]) - 1:
                    phoneme_end = phoneme_end + 1

                if phoneme_start == phoneme_end:
                    # Padding
                    continue

                note_level_aggregation[i][j] = torch.mean(phoneme_embedding[i,phoneme_start:phoneme_end,:], dim=0)
                phoneme_start = phoneme_end

        score_note_dur = score_note_dur.unsqueeze(2)
        score_note_dur_inv = 1.0 / (score_note_dur + 1.0)

        enc_output = torch.cat((note_level_aggregation, score_note_dur, score_note_dur_inv), dim=2)
        enc_output, _ = self.time_lag_rnn(enc_output)
        enc_output = enc_output.contiguous().transpose(1, 2)
        note_dur_diff = self.time_lag_cnn(enc_output).transpose(1, 2)

        return note_dur_diff

class ScoreFeatureExtractor(nn.Module):
    def __init__(self, total_phoneme_num, device='cpu'):
        super(ScoreFeatureExtractor, self).__init__()
        
        self.device = device

        self.total_phoneme_num = total_phoneme_num

        self.kernel = 5
        self.padding = (self.kernel - 1) // 2

        self.note_duration_predictor = Score_dur_to_note_dur(self.device)

        self.lyrics_encoder = LyricsEncoder()
        self.phoneme_dur_predictor = Phoneme_dur_predictor()

        self.note_level_encoder = NoteEncoder()
        self.note_level_f0_predictor = NoteParameter_predictor()
        self.f0_residual_predictor = F0_predictor()

        self.ppg_to_latent_feature1 = nn.Sequential(
            nn.Conv1d(self.total_phoneme_num - 2, hp.encoder_dim, 1, stride=1, padding=0, dilation=1),
            nn.GroupNorm(16, hp.encoder_dim),
            nn.ReLU(),
            nn.Conv1d(hp.encoder_dim, hp.encoder_dim, self.kernel, stride=1, padding=self.padding, dilation=1),
        )

    def LR(self, x, duration, spec_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration, -1), -1)[0]
        alignment = torch.zeros(duration.size(0), expand_max_len, duration.size(1)).numpy()
        alignment = create_alignment(alignment, duration.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(self.device)

        # print (alignment.shape, x.shape, duration.shape)
        output = alignment @ x
        if spec_max_length:
            output = F.pad(
                output, (0, 0, 0, spec_max_length-output.size(1), 0, 0))
        return output

    def soft_LR(self, x, phoneme_seq, is_multi_phoneme, note_duration, duration, spec_max_length=None):
        # duration: (n, 3), each frame has two parameters mean, stddev, amplitude, which mean 
        # , where t is the frame index related to the note offset.

        # note_duration: (m, 1)
        expand_max_len = torch.max(
            torch.sum(note_duration, -1), -1)[0]

        alignment = torch.zeros(duration.size(0), expand_max_len, duration.size(1)).to(self.device)
        output, alignment = create_soft_alignment(x, alignment, phoneme_seq, is_multi_phoneme, note_duration, duration, self.device)

        if spec_max_length:
            output = F.pad(
                output, (0, 0, 0, spec_max_length-output.size(1), 0, 0))

        return output, alignment

    def mask_tensor(self, mel_output, position, mel_max_length, fill_value=0.):
        lengths = torch.max(position, -1)[0]
        mask = ~utils.get_mask_from_lengths(lengths, device=self.device, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, fill_value)

    def get_rel_pos(self, spec_max_length, note_dur):
        rel_pos = torch.zeros(note_dur.size(0), spec_max_length, 4).to(self.device)
        for i in range(len(rel_pos)):
            start = 0
            end = 0

            for j in range(len(note_dur[i])):
                start = end
                end = min(end + note_dur[i][j], spec_max_length)
                rel_pos[i, start:end, 0] = torch.arange(0, end - start, 1).to(self.device)
                rel_pos[i, start:end, 1] = note_dur[i][j] - rel_pos[i, start:end, 0]

                rel_pos[i, start:end, 2] = torch.arange(0, end - start, 1).to(self.device) / note_dur[i][j]
                rel_pos[i, start:end, 3] = 1 - rel_pos[i, start:end, 0]

        return rel_pos

    def expand_note_feature(self, x, duration, spec_max_length=None): # from length regulator
        expand_max_len = torch.max(
            torch.sum(duration, -1), -1)[0]
        alignment = torch.zeros(duration.size(0), expand_max_len, duration.size(1)).numpy()
        alignment = create_alignment(alignment, duration.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(self.device)

        # normalized/unnormalized distance to onset/offset
        alignment_distance = torch.zeros(duration.size(0), expand_max_len, 4).numpy()
        alignment_distance = compute_distance(alignment_distance, duration.cpu().numpy())
        alignment_distance = torch.from_numpy(alignment_distance).to(self.device)

        # print (alignment.shape)
        output = alignment @ x.float()
        if spec_max_length:
            output = F.pad(
                output, (0, 0, 0, spec_max_length-output.size(1), 0, 0))

            alignment_distance = F.pad(
                alignment_distance, (0, 0, 0, spec_max_length-alignment_distance.size(1), 0, 0))
        return output.long(), alignment_distance

    def get_dur_feature(self, word_duration, phoneme_level_note_dur_diff, cum_phoneme_level_note_dur_diff):
        word_duration = word_duration.unsqueeze(2)
        word_duration_inv = 1.0 / (word_duration + 1.0)

        phoneme_level_note_dur_diff = phoneme_level_note_dur_diff.unsqueeze(2)

        phoneme_level_note_dur_diff_pos = torch.clip(phoneme_level_note_dur_diff, min=0)
        phoneme_level_note_dur_diff_neg = -torch.clip(phoneme_level_note_dur_diff, max=0)
        phoneme_level_note_dur_diff_pos_inv = 1.0 / (phoneme_level_note_dur_diff_pos + 1.0)
        phoneme_level_note_dur_diff_neg_inv = 1.0 / (phoneme_level_note_dur_diff_neg + 1.0)

        cum_phoneme_level_note_dur_diff = cum_phoneme_level_note_dur_diff.unsqueeze(2)

        cum_phoneme_level_note_dur_diff_pos = torch.clip(cum_phoneme_level_note_dur_diff, min=0)
        cum_phoneme_level_note_dur_diff_neg = -torch.clip(cum_phoneme_level_note_dur_diff, max=0)
        cum_phoneme_level_note_dur_diff_pos_inv = 1.0 / (cum_phoneme_level_note_dur_diff_pos + 1.0)
        cum_phoneme_level_note_dur_diff_neg_inv = 1.0 / (cum_phoneme_level_note_dur_diff_neg + 1.0)

        duration_feature = torch.cat((word_duration, word_duration_inv
                                        , phoneme_level_note_dur_diff, phoneme_level_note_dur_diff_pos_inv, phoneme_level_note_dur_diff_neg_inv
                                        , cum_phoneme_level_note_dur_diff, cum_phoneme_level_note_dur_diff_pos_inv, cum_phoneme_level_note_dur_diff_neg_inv), dim=2)

        return duration_feature

    def forward(self, batch, ref_encoding_sap, no_predictor, groundtruth, only_score):
        spec_pos = batch["spec_pos"]
        spec_max_length = batch["spec_max_len"]
        
        # phoneme-level
        phoneme_seq = batch["phoneme_seq"]
        phoneme_tone = batch["phoneme_tone"]
        phoneme_order = batch["phoneme_order"]
        is_multi_phoneme = batch["is_multi_phoneme"]
        word_pos = batch["word_pos"]

        # note-level
        note_pitch = batch["note_pitch"]
        score_note_dur = batch["score_note_dur"]
        
        note_dur_diff_prediction = self.note_duration_predictor(score_note_dur, phoneme_seq, phoneme_order)
        note_dur_diff_prediction = note_dur_diff_prediction.squeeze(2)
        note_dur_prediction = note_dur_diff_prediction + score_note_dur.float()

        if only_score:
            note_dur_diff_prediction = torch.round(note_dur_diff_prediction).long()
            note_dur = note_dur_diff_prediction + score_note_dur
            expand_length = torch.sum(note_dur, -1)
            expand_max_len = int(torch.max(expand_length, -1)[0])
            spec_max_length = expand_max_len

            length_spec = np.array(list())
            for cur_expand_length in expand_length:
                length_spec = np.append(length_spec, int(cur_expand_length))

            spec_pos = list()
            max_spec_len = int(max(length_spec))
            for length_spec_row in length_spec:
                spec_pos.append(np.pad([i+1 for i in range(int(length_spec_row))],
                                      (0, max_spec_len-int(length_spec_row)), 'constant'))
            spec_pos = torch.from_numpy(np.array(spec_pos)).to(self.device)

            # Expand word duration
            batch_size, phoneme_num = phoneme_seq.shape
            note_num = note_dur.shape[1]
            word_duration = torch.zeros((batch_size, phoneme_num), device=self.device)
            phoneme_level_note_dur_diff = torch.zeros((batch_size, phoneme_num), device=self.device)
            cum_phoneme_level_note_dur_diff = torch.zeros((batch_size, phoneme_num), device=self.device)

            for i in range(batch_size):
                count = 0
                phoneme_start = 0
                phoneme_end = 0
                cur_cum_note_dur_diff = 0.0
                for j in range(note_num):
                    # The phoneme id of <sep> is 1
                    while phoneme_start < len(phoneme_seq[i]) - 1 and phoneme_seq[i][phoneme_start] <= 1:
                        phoneme_start = phoneme_start + 1

                    phoneme_end = phoneme_start

                    while phoneme_seq[i][phoneme_end] > 1 and phoneme_end < len(phoneme_seq[i]) - 1:
                        phoneme_end = phoneme_end + 1

                    cur_cum_note_dur_diff = cur_cum_note_dur_diff + note_dur_diff_prediction[i][j] / 200.0

                    for k in range(phoneme_start, phoneme_end):
                        word_duration[i][k] = note_dur[i][j] / 200.0
                        phoneme_level_note_dur_diff[i][k] = note_dur_diff_prediction[i][j] / 200.0
                        cum_phoneme_level_note_dur_diff[i][k] = cur_cum_note_dur_diff

                    phoneme_start = phoneme_end

        else:
            note_dur = batch["note_dur"]
            word_duration = batch["word_duration"].float()
            gt_note_dur_diff = (note_dur.float() - score_note_dur.float()) / 200.0

            expand_max_len = int(torch.max(torch.sum(note_dur, -1), -1)[0])

            # Expand word duration
            batch_size, phoneme_num = phoneme_seq.shape
            note_num = note_dur.shape[1]
            phoneme_level_note_dur_diff = torch.zeros((batch_size, phoneme_num), device=self.device)
            cum_phoneme_level_note_dur_diff = torch.zeros((batch_size, phoneme_num), device=self.device)

            for i in range(batch_size):
                count = 0
                phoneme_start = 0
                phoneme_end = 0
                cur_cum_note_dur_diff = 0.0
                for j in range(note_num):
                    # The phoneme id of <sep> is 1
                    while phoneme_start < len(phoneme_seq[i]) - 1 and phoneme_seq[i][phoneme_start] <= 1:
                        phoneme_start = phoneme_start + 1

                    phoneme_end = phoneme_start

                    while phoneme_seq[i][phoneme_end] > 1 and phoneme_end < len(phoneme_seq[i]) - 1:
                        phoneme_end = phoneme_end + 1

                    cur_cum_note_dur_diff = cur_cum_note_dur_diff + gt_note_dur_diff[i][j]

                    for k in range(phoneme_start, phoneme_end):
                        phoneme_level_note_dur_diff[i][k] = gt_note_dur_diff[i][j]
                        cum_phoneme_level_note_dur_diff[i][k] = cur_cum_note_dur_diff

                    phoneme_start = phoneme_end

        # Expand and get frame-level note pitch features
        note_pitch_expanded, alignment_distance = self.expand_note_feature(note_pitch.unsqueeze(2)
            , note_dur, spec_max_length=spec_max_length)
        note_pitch_expanded = note_pitch_expanded.squeeze(2)

        duration_feature = self.get_dur_feature(word_duration, phoneme_level_note_dur_diff, cum_phoneme_level_note_dur_diff)

        # lyrics_encoder_output shape: (n, l, 2*encoder_dim)
        lyrics_encoder_output, phoneme_embedding, ref_phoneme_emb = self.lyrics_encoder(phoneme_seq, ref_encoding_sap, phoneme_tone
            , duration_feature, word_pos)
        soft_pred_duration = self.phoneme_dur_predictor(lyrics_encoder_output, phoneme_embedding, ref_phoneme_emb, phoneme_order, duration_feature)

        one_hot_phoneme = F.one_hot(phoneme_seq, num_classes=self.total_phoneme_num).float().to(self.device)
        one_hot_phoneme_2 = F.one_hot(torch.clip(phoneme_seq + 1, max=self.total_phoneme_num - 1).long()
                                , num_classes=self.total_phoneme_num).float().to(self.device)

        # Because the phoneme id of the phoneme in the same initial/final are adjacent 
        one_hot_phoneme = torch.cat((one_hot_phoneme, one_hot_phoneme_2), dim=2)

        length_regulator_output, alignment = self.soft_LR(one_hot_phoneme, phoneme_seq, is_multi_phoneme, note_dur
            , soft_pred_duration, spec_max_length=spec_max_length)

        length_regulator_output = length_regulator_output.transpose(1, 2)
        length_regulator_output = self.ppg_to_latent_feature1(length_regulator_output[:,2:,:])
        length_regulator_output = length_regulator_output.transpose(1, 2)

        # Generate a note-level pitch embedding (only one embedding layer) for pitch encoder
        # and a note-level pitch feature (CNN) for F0 predictor (can model "transition" between notes)
        note_pitch_embed, note_pitch_cnn_enc = self.note_level_encoder(note_pitch, ref_encoding_sap)
        
        # Expand the two feature vectors
        note_pitch_enc_expanded = self.LR(note_pitch_cnn_enc, note_dur, spec_max_length=spec_max_length)

        f0_parameter = self.note_level_f0_predictor(note_pitch_cnn_enc)
        f0_parameter_expanded = self.LR(f0_parameter, note_dur, spec_max_length=spec_max_length)

        vib_phase_shift = torch.rand(note_dur.shape).unsqueeze(2).to(self.device)
        vib_phase_shift = self.LR(vib_phase_shift, note_dur, spec_max_length=spec_max_length).squeeze(2)

        f0_residual_output, f0_feature, _, _, _ = self.f0_residual_predictor(note_pitch_enc_expanded
            , alignment_distance, f0_parameter_expanded, vib_phase_shift)
        
        f0_predict = self.mask_tensor((f0_residual_output + note_pitch_expanded.unsqueeze(2)), spec_pos, spec_max_length)
        
        return length_regulator_output, f0_predict, alignment, note_dur_prediction, expand_max_len, spec_pos


class Extractor(nn.Module):
    """Convolutional Transformer Decoder Layer"""
    def __init__(
        self, d_model: int, nhead: int, d_hid: int, dropout=0.1, no_residual=False, device='cpu',
    ):
        super(Extractor, self).__init__()

        self.device = device
        self.d_model = d_model

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=0)

        self.dim_reshape = nn.Linear(d_model*2, d_model)

        self.conv1 = nn.Conv1d(d_model, d_hid, 9, padding=4)
        self.conv2 = nn.Conv1d(d_hid, d_model, 1, padding=0)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout2d(dropout)

        self.no_residual = no_residual

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        orig_tgt: torch.Tensor,
        ref_encoding_sap_repeat: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # multi-head self attention

        # src: (frame, batch, feature dim)
        # for dropout: (batch, feature dim (channel), frame)

        tgt_concated = torch.cat((tgt, ref_encoding_sap_repeat), dim=2)
        tgt = self.dim_reshape(tgt_concated)

        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]

        # add & norm
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # multi-head cross attention
        tgt2, attn = self.cross_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )

        # add & norm
        if self.no_residual:
            # tgt = self.dropout2(tgt2)
            tgt = tgt2
        else:
            # tgt = tgt + self.dropout2(tgt2)
            tgt = tgt + tgt2
        tgt = self.norm2(tgt)

        # conv1d
        tgt2 = tgt.transpose(0, 1).transpose(1, 2)
        tgt2 = self.conv2(F.relu(self.conv1(tgt2)))
        tgt2 = self.dropout2(tgt2)
        tgt2 = tgt2.transpose(1, 2).transpose(0, 1)

        tgt = tgt + tgt2
        tgt = self.norm3(tgt)

        return tgt, attn


def pitch_sinusoid_embedding(pitch_value, d_hid, device):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / torch.pow(100, 2 * hid_idx / d_hid)

    hid_idx = torch.tensor([i for i in range(d_hid // 2)] + [i for i in range(d_hid // 2)]).to(device)
    position = pitch_value.unsqueeze(2)
    position = torch.repeat_interleave(position, d_hid, dim=2)

    sinusoid_table = cal_angle(position, hid_idx)

    sinusoid_table[:,:,:d_hid//2] = torch.sin(sinusoid_table[:,:,:d_hid//2])
    sinusoid_table[:,:,d_hid//2:] = torch.cos(sinusoid_table[:,:,d_hid//2:])

    return sinusoid_table.float()

class Pitch_contour_encoding(nn.Module):
    def __init__(self, filter_size, device):
        super(Pitch_contour_encoding, self).__init__()

        self.filter_size = filter_size
        self.device = device

        self.pitch_layer1 = nn.Sequential(
            nn.Linear(self.filter_size, self.filter_size),
            nn.ReLU(),
            nn.Linear(self.filter_size, self.filter_size),
        )

        self.layer_norm1 = nn.LayerNorm(self.filter_size)

        self.pitch_layer2 = nn.Sequential(
            nn.Linear(self.filter_size, self.filter_size),
            nn.ReLU(),
            nn.Linear(self.filter_size, self.filter_size),
        )

        self.layer_norm2 = nn.LayerNorm(self.filter_size)

    def forward(self, pitch_value, target_length):
        pitch_value = pitch_value[:,:target_length].detach()

        pitch_emb = pitch_sinusoid_embedding(pitch_value, self.filter_size, self.device)
        pitch_emb = pitch_emb.contiguous()

        out = self.pitch_layer1(pitch_emb)
        out = self.layer_norm1(out)
        out = self.pitch_layer2(out)
        out = self.layer_norm2(out)
        return out

class AutoSVS(nn.Module):
    """ FastSpeech """
    def __init__(self, device='cpu', phoneme_num=95):
        super(AutoSVS, self).__init__()
        
        self.device = device
        self.phoneme_num = phoneme_num

        # Mel feature extractor for SVC
        self.audio_content_encoder = MelFeatureExtractor(self.device, phoneme_num=phoneme_num)

        self.target_feature_extractor = MelTargetFeatureExtractor(self.device)

        # Score to frame-level features
        self.score_feature_extractor = ScoreFeatureExtractor(total_phoneme_num=phoneme_num, device=self.device)

        # Float pitch -> encoding, shared across both tasks
        self.pitch_contour_encoder = Pitch_contour_encoding(hp.encoder_dim, self.device)

        self.energy_predictor = Energy_predictor()

        self.kernel = 5
        self.padding = (self.kernel - 1) // 2
        
        self.cross_attn1 = Extractor(d_model=hp.encoder_dim, nhead=4, d_hid=hp.encoder_dim, dropout=0.1, device=self.device)
        self.cross_attn2 = Extractor(d_model=hp.encoder_dim, nhead=4, d_hid=hp.encoder_dim, dropout=0.1, device=self.device)
        self.cross_attn3 = Extractor(d_model=hp.encoder_dim, nhead=4, d_hid=hp.encoder_dim, dropout=0.1, device=self.device)

        self.ppg_to_latent_feature_share1 = nn.Sequential(
            nn.Conv1d(hp.encoder_dim, hp.encoder_dim, self.kernel, stride=1, padding=self.padding, dilation=1),
            nn.GroupNorm(16, hp.encoder_dim),
            nn.ReLU(),
            nn.Conv1d(hp.encoder_dim, hp.encoder_dim, self.kernel, stride=1, padding=self.padding, dilation=1),
        )

        self.ppg_to_latent_feature_share2 = nn.Sequential(
            nn.Conv1d(hp.encoder_dim, hp.encoder_dim, self.kernel, stride=1, padding=self.padding, dilation=1),
            nn.GroupNorm(16, hp.encoder_dim),
            nn.ReLU(),
            nn.Conv1d(hp.encoder_dim, hp.encoder_dim, self.kernel, stride=1, padding=self.padding, dilation=1),
        )

        # Decoder
        self.decoder = DecoderPostNet(self.device)

    def mask_tensor(self, mel_output, position, mel_max_length, fill_value=0.):
        lengths = torch.max(position, -1)[0]
        mask = ~utils.get_mask_from_lengths(lengths, device=self.device, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, fill_value)


    def conversion(self, spec_input, spec_pos, ref_mel, ref_mel_global, pitch_value, energy_feature, target_length):
        # Feature extractor
        ref_encoding1, ref_encoding2, ref_encoding3, ref_encoding_sap = self.target_feature_extractor(ref_mel, ref_mel_global)
        
        source_linguistic_encoding, lyrics_predict = self.audio_content_encoder(spec_input, ref_encoding_sap)

        source_linguistic_encoding = source_linguistic_encoding.transpose(1, 2)
        source_linguistic_encoding = source_linguistic_encoding + self.ppg_to_latent_feature_share1(source_linguistic_encoding)
        source_linguistic_encoding = source_linguistic_encoding + self.ppg_to_latent_feature_share2(source_linguistic_encoding)
        source_linguistic_encoding = source_linguistic_encoding.transpose(1, 2)
        
        pitch_encoding = self.pitch_contour_encoder(pitch_value, target_length)

        ref_encoding_sap_repeat = torch.repeat_interleave(ref_encoding_sap, source_linguistic_encoding.shape[1], dim=1)
        decoder_input, attn_output_weights1 = self.cross_attn1(source_linguistic_encoding.transpose(0, 1), ref_encoding3.transpose(0, 1)
            , source_linguistic_encoding.transpose(0, 1), ref_encoding_sap_repeat.transpose(0, 1))

        attn_output_weights1 = attn_output_weights1.cpu()
        ref_encoding3 = ref_encoding3.cpu()

        decoder_input, attn_output_weights2 = self.cross_attn2(decoder_input, ref_encoding2.transpose(0, 1)
            , source_linguistic_encoding.transpose(0, 1), ref_encoding_sap_repeat.transpose(0, 1))

        attn_output_weights2 = attn_output_weights2.cpu()
        ref_encoding2 = ref_encoding2.cpu()

        decoder_input, attn_output_weights3 = self.cross_attn3(decoder_input, ref_encoding1.transpose(0, 1)
            , source_linguistic_encoding.transpose(0, 1), ref_encoding_sap_repeat.transpose(0, 1))

        attn_output_weights3 = attn_output_weights3.cpu()
        ref_encoding1 = ref_encoding1.cpu()
        
        decoder_input = decoder_input.transpose(0, 1)

        decoder_output, decoder_output_after = self.decoder(decoder_input, pitch_encoding, ref_encoding_sap, energy_feature)

        decoder_output = self.mask_tensor(decoder_output, spec_pos, target_length, fill_value=-10.0)
        decoder_output_after = self.mask_tensor(decoder_output_after, spec_pos, target_length, fill_value=-10.0)

        target_attn_output_weights_list = [attn_output_weights1, attn_output_weights2, attn_output_weights3]

        return decoder_output, decoder_output_after, pitch_encoding, lyrics_predict, target_attn_output_weights_list


    def forward(self, batch, alpha=1.0, ret_ref=False, only_encoder=False, no_predictor=False, groundtruth=None, gt_forcing=False, forcing_ratio=0.5, only_score=True
            , pre_computed_target=False, target_encoding=None, use_cycle_consistency=False):

        spec_pos = batch["spec_pos"]
        target_length = batch["spec_max_len"]

        ref_mel = batch["ref_mel"]
        ref_mel_global = batch["ref_mel_global"]

        # ref spec/mel pos
        mel_pos = batch["mel_pos"]

        if pre_computed_target:
            ref_encoding1 = target_encoding[0].to(self.device)
            ref_encoding2 = target_encoding[1].to(self.device)
            ref_encoding3 = target_encoding[2].to(self.device)
            ref_encoding_sap = target_encoding[3].to(self.device)
        else:
            ref_encoding1, ref_encoding2, ref_encoding3, ref_encoding_sap = self.target_feature_extractor(ref_mel, ref_mel_global)

        # ScoreFeatureExtractor
        length_regulator_output, f0_predict, alignment, note_dur_prediction, expand_max_len, spec_pos_refined = self.score_feature_extractor(batch, ref_encoding_sap, no_predictor
                                                                                    , groundtruth, only_score)

        if only_score:
            target_length = expand_max_len
            spec_pos = spec_pos_refined

        length_regulator_output = length_regulator_output.transpose(1, 2)
        length_regulator_output = length_regulator_output + self.ppg_to_latent_feature_share1(length_regulator_output)
        length_regulator_output = length_regulator_output + self.ppg_to_latent_feature_share2(length_regulator_output)
        length_regulator_output = length_regulator_output.transpose(1, 2)

        source_linguistic_encoding = length_regulator_output

        if gt_forcing:
            pitch_gt = groundtruth["pitch"][:,:,0]
            for_pitch_encoder = f0_predict.squeeze(2) * (1.0 - forcing_ratio) + pitch_gt * forcing_ratio
        else:
            for_pitch_encoder = f0_predict.squeeze(2)

        pitch_encoding = self.pitch_contour_encoder(for_pitch_encoder, target_length)
        energy_predict = self.energy_predictor(source_linguistic_encoding, pitch_encoding, ref_encoding_sap)

        if gt_forcing:
            energy_gt = groundtruth["energy"].unsqueeze(2)
            for_energy_decoder = energy_predict * (1.0 - forcing_ratio) + energy_gt * forcing_ratio
        else:
            for_energy_decoder = energy_predict

        ref_encoding_sap_repeat = torch.repeat_interleave(ref_encoding_sap, source_linguistic_encoding.shape[1], dim=1)
        decoder_input, attn_output_weights1 = self.cross_attn1(source_linguistic_encoding.transpose(0, 1), ref_encoding3.transpose(0, 1)
            , source_linguistic_encoding.transpose(0, 1), ref_encoding_sap_repeat.transpose(0, 1))

        attn_output_weights1 = attn_output_weights1.cpu()
        ref_encoding3 = ref_encoding3.cpu()

        decoder_input, attn_output_weights2 = self.cross_attn2(decoder_input, ref_encoding2.transpose(0, 1)
            , source_linguistic_encoding.transpose(0, 1), ref_encoding_sap_repeat.transpose(0, 1))

        attn_output_weights2 = attn_output_weights2.cpu()
        ref_encoding2 = ref_encoding2.cpu()

        decoder_input, attn_output_weights3 = self.cross_attn3(decoder_input, ref_encoding1.transpose(0, 1)
            , source_linguistic_encoding.transpose(0, 1), ref_encoding_sap_repeat.transpose(0, 1))

        attn_output_weights3 = attn_output_weights3.cpu()
        ref_encoding1 = ref_encoding1.cpu()

        decoder_input = decoder_input.transpose(0, 1)
        
        if only_encoder:
            return decoder_input

        decoder_output, decoder_output_after = self.decoder(decoder_input, pitch_encoding, ref_encoding_sap, for_energy_decoder)
        decoder_output = self.mask_tensor(decoder_output, spec_pos, target_length, fill_value=-10.0)
        decoder_output_after = self.mask_tensor(decoder_output_after, spec_pos, target_length, fill_value=-10.0)

        # Output linear spec for spec loss, phoneme alignment for phoneme alignment loss, F0 prediction for F0 loss.
        output = {}
        output['mel'] = decoder_output_after
        output['mel_0'] = decoder_output
        output['alignment'] = alignment
        output['pitch'] = f0_predict.squeeze(2)
        output['note_dur_prediction'] = note_dur_prediction
        output['attn_output_weights'] = [attn_output_weights1, attn_output_weights2, attn_output_weights3]
        output['target_encoding'] = [ref_encoding1, ref_encoding2, ref_encoding3, ref_encoding_sap]
        output['energy'] = energy_predict

        return output, pitch_encoding



class CNNASVmodel(nn.Module):
    def __init__(self, device='cpu'):
        super(CNNASVmodel, self).__init__()
        self.device = device
        self.dnn = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
        )

    def forward(self, x):
        final_feature = self.dnn(x)
        return final_feature