import os,re, random
import numpy as np
import sys, pickle
import librosa
import time
import math

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), '../share'))
from utils import process_text, pad_1D, pad_2D, to_device, load_and_process_dataset, get_param_num

from unlabeled_ctc_dataset import ctc_dataset_collate, LyricsDataset

import hparams as hp

import transformer.Constants as Constants
from transformer.Layers import FFTBlock, PreNet, PostNet, Linear

from model import AutoSVS

from tqdm import tqdm

from dataset import pad_log_2D_tensor, collate_fn_note

from collections import OrderedDict
from typing import Iterator, Tuple


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)

def define_models(device, phoneme_num):
    print ("Phoneme number:", phoneme_num)
    model = AutoSVS(device=device, phoneme_num=phoneme_num).to(device)
    model.apply(init_weights)

    print("Model Has Been Defined")
    num_param = get_param_num(model)
    print('Number of whole model Parameters:', num_param)

    num_param = get_param_num(model.audio_content_encoder)
    print('Number of audio_content_encoder (mel encoder) Parameters:', num_param)

    num_param = get_param_num(model.score_feature_extractor)
    print('Number of score_feature_extractor (score to frame-level feature) Parameters:', num_param)

    return model

def load_models(model, svs_model_path, device):

    print ("Restoring pretrained models (if any)")
    if svs_model_path is not None:
        checkpoint = torch.load(svs_model_path, map_location=device)
        new_state_dict = OrderedDict()

        # if 'decoder.dec_net.8' not in key:
        #     new_state_dict[key] = value

        for key, value in checkpoint.items():
            # print (key)
            if '_noblank' in key:
                key = key.replace('_noblank', '')

            elif 'target_feature_extractor.layer_group1_reshape' in key:
                key = key.replace('target_feature_extractor.layer_group1_reshape', 'target_feature_extractor.reshape_feature')

            elif 'target_feature_extractor.layer_group1_1' in key:
                key = key.replace('target_feature_extractor.layer_group1_1', 'target_feature_extractor.layer_group1.conv1')

            elif 'target_feature_extractor.layer_group1_2' in key:
                key = key.replace('target_feature_extractor.layer_group1_2', 'target_feature_extractor.layer_group1.conv2')

            elif 'target_feature_extractor.layer_group1_3' in key:
                key = key.replace('target_feature_extractor.layer_group1_3', 'target_feature_extractor.layer_group1.conv_downsample')

            elif 'target_feature_extractor.layer_group2_1' in key:
                key = key.replace('target_feature_extractor.layer_group2_1', 'target_feature_extractor.layer_group2.conv1')

            elif 'target_feature_extractor.layer_group2_2' in key:
                key = key.replace('target_feature_extractor.layer_group2_2', 'target_feature_extractor.layer_group2.conv2')

            elif 'target_feature_extractor.layer_group2_3' in key:
                key = key.replace('target_feature_extractor.layer_group2_3', 'target_feature_extractor.layer_group2.conv_downsample')

            elif 'target_feature_extractor.layer_group2_4' in key:
                key = key.replace('target_feature_extractor.layer_group2_4', 'target_feature_extractor.layer_group2.gru')

            elif 'target_feature_extractor.layer_group3_1' in key:
                key = key.replace('target_feature_extractor.layer_group3_1', 'target_feature_extractor.layer_group3.conv1')

            elif 'target_feature_extractor.layer_group3_2' in key:
                key = key.replace('target_feature_extractor.layer_group3_2', 'target_feature_extractor.layer_group3.conv2')

            elif 'target_feature_extractor.layer_group3_3' in key:
                key = key.replace('target_feature_extractor.layer_group3_3', 'target_feature_extractor.layer_group3.conv_downsample')

            elif 'target_feature_extractor.layer_group3_4' in key:
                key = key.replace('target_feature_extractor.layer_group3_4', 'target_feature_extractor.layer_group3.gru')

            elif 'audio_content_encoder.ppg_to_latent_feature2' in key:
                key = key.replace('audio_content_encoder.ppg_to_latent_feature2', 'ppg_to_latent_feature_share1')

            elif 'audio_content_encoder.ppg_to_latent_feature3' in key:
                key = key.replace('audio_content_encoder.ppg_to_latent_feature3', 'ppg_to_latent_feature_share2')


            new_state_dict[key] = value

        model.load_state_dict(new_state_dict, strict=True)
        print("---SVS Model Restored---", svs_model_path, "\n")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu_id')
    parser.add_argument('-svs', '--svs_model_path', default=None)
    
    args = parser.parse_args()

    gpu_id = args.gpu_id

    svs_model_path = args.svs_model_path

    # Get dataset
    phoneme_list = ['PAD', 'sep', 'm', 'ei', 'c', 'ii', 'w', 'uo', 'z', 'ong', 'y', 'i', 'g', 'e', 'r', 'en', 'ou'
        , 'j', 'iao', 'ch', 'a', 'l', 'u', 'k', 'sh', 'eng', 'h', 'zh', 'n', 'q', 've', 'd', 'ai', 'iou', 'x', 'iang', 't', 'ang'
        , 'ua', 'in', 'ian', 'an', 'v', 'iong', 's', 'uan', 'b', 'f', 'ing', 'ao', 'van', 'o', 'uei', 'p', 'ie', 'iii', 'uen', 'vn'
        , 'uang', 'ia', 'er', 'uai', 'sil', 'io', 'ueng']

    multi_phone_pinyin = ['ei', 'uo', 'ong', 'en', 'ou', 'eng', 've', 'ai', 'ang', 'ua', 'in', 'an', 'ing', 'io', 'ao', 'ie', 'vn', 'ia'
        , 'iao', 'iou', 'iang', 'ian', 'iong', 'uan', 'van', 'uei', 'uen', 'uang', 'uai', 'ueng']

    phoneme_id_list = []
    have_multi_phone = []
    for i in range(len(phoneme_list)):
        phoneme_id_list.append(phoneme_list[i])

        if phoneme_list[i] in multi_phone_pinyin:
            second_phoneme = str(phoneme_list[i]) + '_2'
            phoneme_id_list.append(second_phoneme)
            have_multi_phone.append(True)
            have_multi_phone.append(True)
        else:
            have_multi_phone.append(False)


    print ("Use GPU #", gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')

    # Define model
    model = define_models(device, len(phoneme_id_list))
    print("Defined Optimizer and Loss Function.", time.time())
    # Load checkpoint if exists
    model = load_models(model, svs_model_path, device)
    
    save_dict = model.state_dict()
    target_model_path = 'model_0311_propose_300000.pth.tar'
    torch.save(save_dict, target_model_path)