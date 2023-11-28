import os,re
import collections
import soundfile as sf
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import sys
import h5py
import librosa
from scipy import signal
from multiprocessing import cpu_count
import argparse
import time
import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

#FastSpeech
from utils import process_text, pad_1D, pad_2D, pad_1D_tensor, pad_2D_tensor, pad_log_2D_tensor

import hparams as hp


frame_per_second = 200.0

# def reformulate_phoneme_seq(input_list, phoneme_list, have_multi_phone):
#     cur_phoneme_orders = []
#     cur_phoneme_order = 0

#     new_input_data = []

#     for j in range(len(input_list)):
#         # print (input_data[i][0][j][0])
#         if input_list[j][0] == 'sep':
#             cur_phoneme_order = 0
#         else:
#             cur_phoneme_order = cur_phoneme_order + 1
#         cur_phoneme_orders.append(cur_phoneme_order)

#         # print (input_data[i][0][j])

#         input_list[j][0] = phoneme_list.index(input_list[j][0])
#         is_multi_phone = have_multi_phone[input_list[j][0]]
#         input_list[j].append(is_multi_phone)
#         # "Is the start of an initial/final."
#         input_list[j].append(1)

#         new_input_data.append(list(input_list[j]))

#         if is_multi_phone:
#             new_input_data.append(list(input_list[j]))
#             new_input_data[-1][0] = new_input_data[-1][0] + 1
#             # "Is NOT the start of an initial/final, but the second phone of an initial/final."
#             new_input_data[-1][-1] = 0
#             cur_phoneme_order = cur_phoneme_order + 1
#             cur_phoneme_orders.append(cur_phoneme_order)

#         # print (input_data[i][0][j])
#     return new_input_data, cur_phoneme_orders

def reformulate_phoneme_seq(input_list, phoneme_list, have_multi_phone):
    cur_phoneme_orders = []
    cur_phoneme_order = 0

    for j in range(len(input_list)):
        if input_list[j][0] == 'sep':
            cur_phoneme_order = 0
        else:
            cur_phoneme_order = cur_phoneme_order + 1
        cur_phoneme_orders.append(cur_phoneme_order)

        input_list[j][0] = phoneme_list.index(input_list[j][0])
        is_multi_phone = have_multi_phone[input_list[j][0]]
        input_list[j].append(is_multi_phone)

    return input_list, cur_phoneme_orders

class MpopDataset(Dataset):

    # def __init__(self, gt, input_data, note_data, lookup_table, waveforms, spk_embedding, local_embedding, phoneme_list, have_multi_phone, is_test=False):
    def __init__(self, gt, input_data, note_data, lookup_table, waveforms, phoneme_list, have_multi_phone, is_test=False):
        
        phoneme_order_in_note = []

        for i in range(len(input_data)):
            # print (input_data[i])
            input_data[i][0], cur_phoneme_orders = reformulate_phoneme_seq(input_data[i][0], phoneme_list, have_multi_phone)
            input_data[i][0] = np.array(input_data[i][0])
            phoneme_order_in_note.append(np.array(cur_phoneme_orders))

            # print (input_data[i][0])
            # print (cur_phoneme_orders)


        for i in range(len(gt)):
            for j in range(len(gt[i])):
                gt[i][j] = np.array(gt[i][j])

        # print (phoneme_list)

        # process note data, convert duration to frame number.
        # [Note pitch, note duration, score note duration]
        for i in range(len(note_data)):
            last_frame = 0
            last_time = 0
            for j in range(len(note_data[i])):
                last_time = last_time + note_data[i][j][1]
                cur_frame = int(round(last_time * frame_per_second))
                note_data[i][j][1] = cur_frame - last_frame
                last_frame = cur_frame

        for i in range(len(note_data)):
            last_frame = 0
            last_time = 0
            for j in range(len(note_data[i])):
                last_time = last_time + note_data[i][j][2]
                cur_frame = int(round(last_time * frame_per_second))
                note_data[i][j][2] = cur_frame - last_frame
                last_frame = cur_frame

        self.gt = gt
        self.input_data = input_data
        self.note_data = note_data
        # self.ref = ref
        self.lookup_table = lookup_table
        self.phoneme_list = phoneme_list
        self.phoneme_order_in_note = phoneme_order_in_note
        self.waveforms = waveforms
        self.have_multi_phone = have_multi_phone
        # self.spk_embedding = spk_embedding
        # self.local_embedding = local_embedding

        self.is_test = is_test

        self.decrease_steps = 10000.0
        self.total_step = 0
        
    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        # same singer, different ref audio
        # gt: [gt_mel, gt_energy, gt_pitch, gt_spec, gt_phoneme_duration]
        # input_data: [cur_segments_input, singer_name]

        # self.use_self_prob = 1.0 - self.total_step / self.decrease_steps

        # ran_value = torch.rand(1)
        # # if self.is_test == True or ran_value > self.use_self_prob:
        # if self.is_test == True:
        #     singer_id = self.input_data[idx][1]
        #     ref_idx = torch.randint(low=0, high=len(self.lookup_table[singer_id]), size=(10,))

        #     ref_lookup_idx = [self.lookup_table[singer_id][int(ref_idx[i])] for i in range(len(ref_idx))]
        #     ref_mel = np.concatenate([self.gt[ref_lookup_idx[i]][0] for i in range(len(ref_lookup_idx))], axis=0)
        #     spk_embedding = torch.cat([self.spk_embedding[ref_lookup_idx[i]] for i in range(len(ref_lookup_idx))], dim=0)
        #     spk_embedding = torch.mean(spk_embedding, dim=0, keepdim=True)

        #     # print (ref_mel.shape, spk_embedding.shape)
            
        #     return {'mel': self.gt[idx][0],
        #             'energy': self.gt[idx][1][:,0], # squeeze
        #             'pitch': self.gt[idx][2], #[n, 2]
        #             'dur': self.gt[idx][3],
        #             'waveforms': self.waveforms[idx],
        #             'input_feature': self.input_data[idx][0], #[phoneme->one-hot, tone, note duration, note_pitch]
        #             'target_singer': self.input_data[idx][1],
        #             'phoneme_order': self.phoneme_order_in_note[idx],
        #             'ref_mel': ref_mel,
        #             'note_data': self.note_data[idx],
        #             # 'spk_embedding':self.spk_embedding[ref_lookup_idx]
        #             'spk_embedding': spk_embedding
        #             }

        # else:
        #     # ref_mel = np.concatenate([self.gt[idx][0] for i in range(10)], axis=0)
        #     ref_mel = self.gt[idx][0]
            
        #     # print (ref_mel.shape)
        #     return {'mel': self.gt[idx][0],
        #             'energy': self.gt[idx][1][:,0], # squeeze
        #             'pitch': self.gt[idx][2], #[n, 2]
        #             'dur': self.gt[idx][3],
        #             'waveforms': self.waveforms[idx],
        #             'input_feature': self.input_data[idx][0], #[phoneme->one-hot, tone, note duration, note_pitch]
        #             'target_singer': self.input_data[idx][1],
        #             'phoneme_order': self.phoneme_order_in_note[idx],
        #             # 'ref_mel': self.gt[idx][0],
        #             'ref_mel': ref_mel,
        #             # 'ref_spec': self.gt[idx][3],
        #             'note_data': self.note_data[idx],
        #             'spk_embedding':self.spk_embedding[idx]
        #             }


        self.use_self_prob = 1.0 - self.total_step / self.decrease_steps

        self.ref_sample_method = 'append'

        ran_value = torch.rand(1)
        if self.is_test == True:
        # if self.is_test == True:
            singer_id = self.input_data[idx][1]
            ref_idx = torch.randint(low=0, high=len(self.lookup_table[singer_id]), size=(10,))

            ref_lookup_idx = [self.lookup_table[singer_id][int(ref_idx[i])] for i in range(len(ref_idx))]
            ref_mel = np.concatenate([self.gt[ref_lookup_idx[i]][0] for i in range(len(ref_lookup_idx))], axis=0)
            # spk_embedding = torch.cat([self.spk_embedding[ref_lookup_idx[i]] for i in range(len(ref_lookup_idx))], dim=0)
            # spk_embedding = torch.mean(spk_embedding, dim=0, keepdim=True)

            # print (ref_mel.shape, spk_embedding.shape)
            
            return {'mel': self.gt[idx][0],
                    'energy': self.gt[idx][1][:,0], # squeeze
                    'pitch': self.gt[idx][2], #[n, 2]
                    'dur': self.gt[idx][3],
                    'waveforms': self.waveforms[idx],
                    'input_feature': self.input_data[idx][0], #[phoneme->one-hot, tone, note duration, note_pitch, score note duration]
                    'target_singer': self.input_data[idx][1],
                    'phoneme_order': self.phoneme_order_in_note[idx],
                    'ref_mel': ref_mel,
                    'note_data': self.note_data[idx],
                    # 'spk_embedding':self.spk_embedding[ref_lookup_idx]
                    # 'spk_embedding': spk_embedding
                    }
        else:
            if self.ref_sample_method == 'append':
                singer_id = self.input_data[idx][1]
                # Take the source audio clip and 9 other audio clips
                ref_idx = torch.randint(low=0, high=len(self.lookup_table[singer_id]), size=(4,))

                ref_lookup_idx = [self.lookup_table[singer_id][int(ref_idx[i])] for i in range(len(ref_idx))]

                ref_mel = []
                for i in range(len(ref_lookup_idx)):
                    ref_mel.append(self.gt[ref_lookup_idx[i]][0])
                    ref_mel.append(np.full((100, self.gt[ref_lookup_idx[i]][0].shape[-1]), fill_value=-10.0))

                ref_mel.append(self.gt[idx][0])
                ref_mel = np.concatenate(ref_mel, axis=0)
            else:
                singer_id = self.input_data[idx][1]
                # Take the source audio clip and 9 other audio clips
                ref_idx = torch.randint(low=0, high=len(self.lookup_table[singer_id]), size=(5,))

                ref_lookup_idx = [self.lookup_table[singer_id][int(ref_idx[i])] for i in range(len(ref_idx))]

                ref_mel = []
                for i in range(len(ref_lookup_idx)):
                    ref_mel.append(self.gt[ref_lookup_idx[i]][0])
                    ref_mel.append(np.full((100, self.gt[ref_lookup_idx[i]][0].shape[-1]), fill_value=-10.0))

                # ref_mel.append(self.gt[idx][0])
                ref_mel = np.concatenate(ref_mel, axis=0)

            # spk_embedding = [self.spk_embedding[ref_lookup_idx[i]] for i in range(len(ref_lookup_idx))]
            # spk_embedding.append(self.spk_embedding[idx])

            # spk_embedding = torch.cat(spk_embedding, dim=0)
            # spk_embedding = torch.mean(spk_embedding, dim=0, keepdim=True)


            # ref_mel = self.gt[idx][0]
            # spk_embedding = self.spk_embedding[idx]
            
            # print (ref_mel.shape)
            return {'mel': self.gt[idx][0],
                    'energy': self.gt[idx][1][:,0], # squeeze
                    'pitch': self.gt[idx][2], #[n, 2]
                    'dur': self.gt[idx][3],
                    'waveforms': self.waveforms[idx],
                    'input_feature': self.input_data[idx][0], #[phoneme->one-hot, tone, note duration, note_pitch]
                    'target_singer': self.input_data[idx][1],
                    'phoneme_order': self.phoneme_order_in_note[idx],
                    # 'ref_mel': self.gt[idx][0],
                    'ref_mel': ref_mel,
                    # 'ref_spec': self.gt[idx][3],
                    'note_data': self.note_data[idx],
                    # 'spk_embedding':spk_embedding
                    }

  
def reprocess_tensor_note(batch, cut_list):
    # masking padded frames
    # for note-level input
    input_features = [batch[ind]['input_feature'] for ind in cut_list]
    phoneme_order = [batch[ind]['phoneme_order'] for ind in cut_list]

    length_word = np.array([])
    for word in input_features:
        # print (word.shape)
        length_word = np.append(length_word, word.shape[0])

    word_pos = list()
    max_len = int(max(length_word))
    for length_src_row in length_word:
        word_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    word_pos = torch.from_numpy(np.array(word_pos))

    input_features = pad_2D_tensor(input_features)
    phoneme_order = pad_1D_tensor(phoneme_order)


    # for output spectrogram
    specs = [batch[ind]['mel'] for ind in cut_list]
    length_spec = np.array(list())
    for spec in specs:
        length_spec = np.append(length_spec, spec.shape[0])

    spec_pos = list()
    max_spec_len = int(max(length_spec))
    for length_spec_row in length_spec:
        spec_pos.append(np.pad([i+1 for i in range(int(length_spec_row))],
                              (0, max_spec_len-int(length_spec_row)), 'constant'))
    spec_pos = torch.from_numpy(np.array(spec_pos))

    ref_mel = [batch[ind]['ref_mel'] for ind in cut_list]
    length_ref_mel = np.array(list())
    for ref_mel_data in ref_mel:
        length_ref_mel = np.append(length_ref_mel, ref_mel_data.shape[0])

    mel_pos = list()
    max_mel_len = int(max(length_ref_mel))
    for length_ref_mel_row in length_ref_mel:
        mel_pos.append(np.pad([i+1 for i in range(int(length_ref_mel_row))],
                              (0, max_mel_len-int(length_ref_mel_row)), 'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))

    ref_mel = pad_log_2D_tensor(ref_mel, pad_value=-10.0)


    mel = [batch[ind]['mel'] for ind in cut_list]
    mel = pad_log_2D_tensor(mel, pad_value=-10.0)

    pitch = [batch[ind]['pitch'] for ind in cut_list]
    pitch = pad_log_2D_tensor(pitch, pad_value=0)

    dur = [batch[ind]['dur'] for ind in cut_list]
    dur = pad_1D_tensor(dur)

    waveforms = [batch[ind]['waveforms'] for ind in cut_list]
    waveforms = pad_1D_tensor(waveforms)

    energy = [batch[ind]['energy'] for ind in cut_list]
    energy = pad_1D_tensor(energy, PAD=-10.0)

    note_data = [np.array(batch[ind]['note_data']) for ind in cut_list]
    note_data = pad_2D_tensor(note_data)

    note_pitch = note_data[:,:,0]
    note_dur = note_data[:,:,1]
    score_note_dur = note_data[:,:,2]

    # soft alignment gt
    expand_max_len = torch.max(
            torch.sum(note_dur, -1), -1)[0]
    gt_alignment = torch.zeros(len(dur), expand_max_len, len(dur[0]))
    for i in range(len(dur)):
        cur_dur = 0
        last_frame = 0
        total_time = 0
        for j in range(len(dur[i])):
            total_time = total_time + float(dur[i][j])
            cur_frame = int(round(total_time * frame_per_second))

            for k in range(last_frame, min(cur_frame, len(gt_alignment[i]) - 1)):
                gt_alignment[i][k][j] = 1.0
            last_frame = cur_frame

    target_singer = [batch[ind]['target_singer'] for ind in cut_list]

    out = {"input": {'input_feature': input_features,
                       'phoneme_order': phoneme_order,
                       'word_pos': word_pos,
                       'spec_pos': spec_pos,
                       'spec_max_len': max_spec_len,
                       'mel_pos': mel_pos,
                       'max_mel_len': max_mel_len,
                       'ref_mel': ref_mel,
                       'note_pitch': note_pitch,
                       'note_dur': note_dur,
                       'score_note_dur': score_note_dur,
                       },
            "gt":   {'mel': mel,
                       'energy': energy,
                       'pitch': pitch,
                       'dur': dur,
                       'gt_alignment': gt_alignment,
                       'waveforms': waveforms,
                       'target_singer': target_singer}}

    return out


def collate_fn_note(data):
    data_size = len(data)
    # real_batchsize = data_size // hp.batch_expand_size
    real_batchsize = hp.batch_size

    len_arr = np.array([d["input_feature"].shape[0] for d in data])
    idx_arr = np.argsort(-len_arr)

    tail = idx_arr[len(idx_arr) - (len(idx_arr) % real_batchsize) :]
    idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % real_batchsize)]

    idx_arr = idx_arr.reshape((-1, real_batchsize)).tolist()

    if len(tail) > 0:
        idx_arr += [tail.tolist()]

    output = list()
    for idx in idx_arr:
        output.append(reprocess_tensor_note(data, idx))

    # print (output)
    return output
