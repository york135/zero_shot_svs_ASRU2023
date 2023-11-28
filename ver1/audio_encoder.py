import os,re
import numpy as np
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import hparams as hp

class MelFeatureExtractor(nn.Module):
    def __init__(self, device, phoneme_num):
        super(MelFeatureExtractor, self).__init__()
        
        self.device = device

        self.input_size = hp.num_spec
        self.filter_size = 256
        self.output_size = 256


        self.phoneme_classify_size = phoneme_num
        self.kernel = 5
        self.dropout = 0.1

        self.device = device

        self.padding = (self.kernel - 1) // 2


        self.downsample1 = nn.Sequential(
            nn.Conv1d(self.input_size, self.filter_size, self.kernel, stride=1, padding=self.padding, dilation=1),
            nn.GroupNorm(16, self.filter_size),
            nn.ReLU(),
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=2, padding=self.padding, dilation=1),
        )

        self.downsample2 = nn.Sequential(
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding, dilation=1),
            nn.GroupNorm(16, self.filter_size),
            nn.ReLU(),
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=2, padding=self.padding, dilation=1),
            
        )

        self.layer_group1 = nn.Sequential(
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding, dilation=1),
            nn.GroupNorm(16, self.filter_size),
            nn.ReLU(),
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding*2, dilation=2),
            nn.Dropout2d(self.dropout),
        )

        self.layer_group2 = nn.Sequential(
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding, dilation=1),
            nn.GroupNorm(16, self.filter_size),
            nn.ReLU(),
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding*2, dilation=2),
            nn.Dropout2d(self.dropout),
        )


        self.layer_group3 = nn.Sequential(
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding, dilation=1),
            nn.GroupNorm(16, self.filter_size),
            nn.ReLU(),
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding*2, dilation=2),
            nn.Dropout2d(self.dropout),
        )

        self.layer_group4 = nn.Sequential(
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding, dilation=1),
            nn.GroupNorm(16, self.filter_size),
            nn.ReLU(),
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding*2, dilation=2),
            nn.Dropout2d(self.dropout),
            
        )

        self.layer_group_gru = nn.GRU(
            input_size=self.filter_size,
            hidden_size=self.filter_size,
            batch_first=True,
            num_layers=1,
            dropout=0,
            bidirectional=False
        )

        self.upsample1 = nn.Sequential(
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding, dilation=1),
            nn.GroupNorm(16, self.filter_size),
            nn.ReLU(),
            nn.ConvTranspose1d(self.filter_size, self.filter_size, self.kernel, stride=2, padding=self.padding, dilation=1)
        )

        self.upsample2 = nn.Sequential(
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding, dilation=1),
            nn.GroupNorm(16, self.filter_size),
            nn.ReLU(),
            nn.ConvTranspose1d(self.filter_size, self.filter_size, self.kernel, stride=2, padding=self.padding, dilation=1)
        )

        self.phoneme_classifier = nn.Sequential(
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding, dilation=1),
            nn.GroupNorm(16, self.filter_size),
            nn.ReLU(),
            nn.Conv1d(self.filter_size, self.phoneme_classify_size, self.kernel, stride=1, padding=self.padding, dilation=1),
        )

        self.ppg_to_latent_feature1 = nn.Sequential(
            nn.Conv1d(self.phoneme_classify_size - 2, self.filter_size, 1, stride=1, padding=0, dilation=1),
            nn.GroupNorm(16, self.filter_size),
            nn.ReLU(),
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding, dilation=1),
        )


    def forward(self, spec_input, ref_encoding_sap):
        x = spec_input.contiguous().transpose(1, 2)

        spec_shapes = [x.shape[-1],]

        x = self.downsample1(x)
        spec_shapes.append(x.shape[-1])
        x = self.downsample2(x)

        x = x.transpose(1, 2)
        x, _ = self.layer_group_gru(x)
        x = x.transpose(1, 2)

        x = x + self.layer_group1(x)
        x = x + self.layer_group2(x)
        x = x + self.layer_group3(x)
        x = x + self.layer_group4(x)

        x = self.upsample1(x)
        if x.shape[-1] > spec_shapes[-1]:
            x = x[:,:,spec_shapes[-1]]

        if x.shape[-1] < spec_shapes[-1]:
            x = torch.cat((x, torch.zeros((x.shape[0], x.shape[1], spec_shapes[-1] - x.shape[-1]), device=self.device)), dim=2)
        del spec_shapes[-1]

        x = self.upsample2(x)
        if x.shape[-1] > spec_shapes[-1]:
            x = x[:,:,spec_shapes[-1]]

        if x.shape[-1] < spec_shapes[-1]:
            x = torch.cat((x, torch.zeros((x.shape[0], x.shape[1], spec_shapes[-1] - x.shape[-1]), device=self.device)), dim=2)
        del spec_shapes[-1]

        phoneme_predict = self.phoneme_classifier(x)
        phoneme_predict_log_sm = F.log_softmax(phoneme_predict[:,2:,:], dim=1)
        phoneme_predict_sm = torch.exp(phoneme_predict_log_sm)

        phonetic_ppg = phoneme_predict_sm

        x = self.ppg_to_latent_feature1(phonetic_ppg)
        
        x = x.transpose(1, 2)
        phoneme_predict = phoneme_predict.transpose(1, 2)

        return x, phoneme_predict

class ReferenceEncoderBlock(nn.Module):
    def __init__(self, filter_size, kernel, padding, use_gru, device='cpu'):
        super(ReferenceEncoderBlock, self).__init__()

        self.device = device
        self.filter_size = filter_size
        self.kernel = kernel
        self.padding = padding
        self.use_gru = use_gru

        self.conv1 = nn.Sequential(
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding, dilation=1),
            nn.GroupNorm(16, self.filter_size),
            nn.ReLU(),
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding*2, dilation=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding, dilation=1),
            nn.GroupNorm(16, self.filter_size),
            nn.ReLU(),
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding*2, dilation=2),
        )

        self.conv_downsample = nn.Sequential(
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=2, padding=self.padding, dilation=1),
        )

        if self.use_gru:
            self.gru = nn.GRU(
                input_size=self.filter_size,
                hidden_size=self.filter_size,
                batch_first=True,
                num_layers=1,
                dropout=0,
                bidirectional=False
            )

    def forward(self, x):
        x = x + self.conv1(x)
        x = x + self.conv2(x)
        x = self.conv_downsample(x)

        if self.use_gru:
            x = x.transpose(1, 2)
            x, _ = self.gru(x)
            x = x.transpose(1, 2)
        
        return x


class MelTargetFeatureExtractor(nn.Module):
    def __init__(self, device='cpu'):
        super(MelTargetFeatureExtractor, self).__init__()
        
        self.device = device

        self.input_size = hp.num_spec
        self.filter_size = 256
        self.output_size = 256
        self.kernel = 5
        self.dropout = 0

        self.padding = (self.kernel - 1) // 2

        self.channel_expansion = nn.Sequential(
            nn.Conv1d(self.input_size, self.filter_size, self.kernel, stride=1, padding=self.padding, dilation=1),
        )

        self.reshape_feature = nn.Sequential(
            nn.Conv1d(self.filter_size * 2, self.filter_size, self.kernel, stride=1, padding=self.padding, dilation=1),
        )

        self.layer_group1 = ReferenceEncoderBlock(filter_size=self.filter_size, 
                                                    kernel=self.kernel, 
                                                    padding=self.padding,
                                                    use_gru=False, 
                                                    device=self.device)

        self.mask_frame1 = nn.Dropout2d(self.dropout)
        self.content_embedding1 = nn.Parameter(torch.rand((1, 200, self.filter_size), dtype=torch.float32, requires_grad=True))
        self.content_attn1 = nn.MultiheadAttention(self.filter_size, 2, dropout=0, batch_first=True)


        self.layer_group2 = ReferenceEncoderBlock(filter_size=self.filter_size, 
                                                    kernel=self.kernel, 
                                                    padding=self.padding,
                                                    use_gru=True, 
                                                    device=self.device)

        self.mask_frame2 = nn.Dropout2d(self.dropout)
        self.content_embedding2 = nn.Parameter(torch.rand((1, 200, self.filter_size), dtype=torch.float32, requires_grad=True))
        self.content_attn2 = nn.MultiheadAttention(self.filter_size, 2, dropout=0, batch_first=True)

        self.layer_group3 = ReferenceEncoderBlock(filter_size=self.filter_size, 
                                                    kernel=self.kernel, 
                                                    padding=self.padding,
                                                    use_gru=True, 
                                                    device=self.device)

        self.mask_frame3 = nn.Dropout2d(self.dropout)
        self.content_embedding3 = nn.Parameter(torch.rand((1, 200, self.filter_size), dtype=torch.float32, requires_grad=True))
        self.content_attn3 = nn.MultiheadAttention(self.filter_size, 2, dropout=0, batch_first=True)

        self.layer_group4_1 = nn.Sequential(
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding, dilation=1),
            nn.GroupNorm(16, self.filter_size),
            nn.ReLU(),
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding*2, dilation=2),
            nn.GroupNorm(16, self.filter_size),
            nn.ReLU(),
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding*4, dilation=4),
        )
        
        self.layer_group4_2 = nn.Sequential(
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding, dilation=1),
            nn.GroupNorm(16, self.filter_size),
            nn.ReLU(),
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding*2, dilation=2),
            nn.GroupNorm(16, self.filter_size),
            nn.ReLU(),
            nn.Conv1d(self.filter_size, self.filter_size, self.kernel, stride=1, padding=self.padding*4, dilation=4),
        )

        self.sap_attn = nn.Sequential(
            nn.Conv1d(self.filter_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, self.filter_size, kernel_size=1),
            nn.Softmax(dim=2),
            )


    def forward(self, spec_input, ref_mel_global):
        x_orig = spec_input.contiguous().transpose(1, 2)

        x = self.channel_expansion(x_orig)

        ref_mel_global_input = ref_mel_global.contiguous().transpose(1, 2)
        ref_mel_global_input = self.channel_expansion(ref_mel_global_input)
        ref_mel_global_input = ref_mel_global_input + self.layer_group4_1(ref_mel_global_input)
        ref_mel_global_input = ref_mel_global_input + self.layer_group4_2(ref_mel_global_input)
        x_4 = ref_mel_global_input

        attn_weight = self.sap_attn(x_4)
        x_sap = torch.sum(x_4 * attn_weight, dim=2).unsqueeze(1)

        x_sap_expand = torch.repeat_interleave(x_sap, x.shape[2], dim=1)
        x_sap_expand = x_sap_expand.transpose(1, 2)

        temp = torch.cat((x, x_sap_expand), dim=1)
        temp = self.reshape_feature(temp)

        temp = self.layer_group1(temp)
        x_1 = self.mask_frame1(temp.contiguous().transpose(1, 2))

        content_attn1_expand = torch.repeat_interleave(self.content_embedding1, x_1.shape[0], dim=0)
        x_1, _ = self.content_attn1(content_attn1_expand, x_1, x_1)
        x_1 = x_1 + content_attn1_expand

        temp = self.layer_group2(temp)
        x_2 = self.mask_frame2(temp.contiguous().transpose(1, 2))

        content_attn2_expand = torch.repeat_interleave(self.content_embedding2, x_2.shape[0], dim=0)
        x_2, _ = self.content_attn2(content_attn2_expand, x_2, x_2)
        x_2 = x_2 + content_attn2_expand

        temp = self.layer_group3(temp)
        x_3 = self.mask_frame3(temp.contiguous().transpose(1, 2))

        content_attn3_expand = torch.repeat_interleave(self.content_embedding3, x_3.shape[0], dim=0)
        x_3, _ = self.content_attn3(content_attn3_expand, x_3, x_3)
        x_3 = x_3 + content_attn3_expand

        return x_1, x_2, x_3, x_sap