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


spec_loss_function = nn.MSELoss()
dur_loss = nn.L1Loss(reduction='none')
pitch_res_loss = nn.L1Loss()
energy_loss_func = nn.L1Loss()
dur_l1_loss = nn.L1Loss()
lyrics_ctc_loss = nn.CTCLoss()

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

def spec_loss_counter(prediction, gt, mask):
    energy_mask = torch.where(mask >= -3.0, 1.0, 0.0)
    total_loss = spec_loss_function(prediction * energy_mask, gt * energy_mask)
    return total_loss

def get_supervised_loss(pred, gt, input_data):
    energy_pred = pred['energy']
    energy_gt = gt['energy'].unsqueeze(2)

    energy_pred_delta = energy_pred[:,1:,:] - energy_pred[:,:-1,:]
    energy_gt_delta = energy_gt[:,1:,:] - energy_gt[:,:-1,:]

    mel_loss = (spec_loss_counter(pred['mel'], gt['mel'], mask=energy_gt) + spec_loss_counter(pred['mel_0'], gt['mel'], mask=energy_gt)) / 2.0
    align_loss = dur_loss(pred['alignment'].cpu(), gt['gt_alignment']) - 0.25
    align_loss = torch.mean(torch.clip(align_loss, min=0)) * 10.0

    pitch_loss = pitch_res_loss(pred['pitch'], gt['pitch'][:,:,0])

    energy_loss = energy_loss_func(energy_pred, energy_gt) + energy_loss_func(energy_pred_delta, energy_gt_delta)

    note_dur_prediction_loss = dur_l1_loss(pred['note_dur_prediction'], gt['note_dur']) / 10.0

    return mel_loss, align_loss, pitch_loss, energy_loss, note_dur_prediction_loss


def load_and_process_ctc_dataset(ctc_dataset_path, phoneme_list, have_multi_phone):
    print (ctc_dataset_path)
    with open(ctc_dataset_path, 'rb') as f:
        train_ctc_dataset = pickle.load(f)

    train_ctc_dataset.enroll_phoneme_list(phoneme_list, have_multi_phone)
    return train_ctc_dataset

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
        for key, value in checkpoint.items():
            new_state_dict[key] = value

        model.load_state_dict(new_state_dict, strict=False)
        print("---SVS Model Restored---", svs_model_path, "\n")

    return model

from torch.optim.lr_scheduler import LambdaLR
def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Borrowed from FragmentVC (https://github.com/yistLin/FragmentVC)
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.2, 0.2 + 0.4 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def unlabeled_ctc_dataset_forward(model, unlabeled_train_iter, device, method, extra_ref_num):
    batch = next(unlabeled_train_iter)

    batch['mel'] = batch['mel'].to(device)
    batch['ref_mel'] = batch['ref_mel'].to(device)
    batch['ref_global_mel'] = batch['ref_global_mel'].to(device)
    batch['pitch'] = batch['pitch'].to(device)
    batch['energy'] = batch['energy'].to(device)

    target_length = batch['mel'].shape[1]

    spec_pos = [[j+1 for j in range(int(target_length))] for i in range(len(batch['mel']))]
    spec_pos = torch.tensor(spec_pos).to(device)

    dec_output, dec_output_post, _, lyrics_pred, _ = model.conversion(batch['mel'], spec_pos
                        , ref_mel=batch['ref_mel'], ref_mel_global=batch['ref_global_mel'], pitch_value=batch['pitch'][:,:,0]
                        , energy_feature=batch['energy'], target_length=target_length)


    reconstruct_loss = (spec_loss_counter(dec_output, batch['mel'], mask=batch['energy']) 
        + spec_loss_counter(dec_output_post, batch['mel'], mask=batch['energy'])) * 0.5

    lyrics_predict_log_sm = F.log_softmax(lyrics_pred, dim=2)
    lyrics_predict_log_sm = lyrics_predict_log_sm.transpose(0, 1)

    gt_lyrics = batch['lyrics_token']
    gt_lyrics = [torch.tensor(gt_lyrics[i]) for i in range(len(gt_lyrics))]

    target_lengths = [len(gt_lyrics[i]) for i in range(len(gt_lyrics))]
    input_lengths = [lyrics_predict_log_sm.shape[0] for i in range(len(gt_lyrics))]

    gt_lyrics_target = torch.cat(gt_lyrics, dim=0).to(device)

    ctc_classify_loss = lyrics_ctc_loss(lyrics_predict_log_sm, gt_lyrics_target
        , input_lengths, target_lengths)

    return (reconstruct_loss, ctc_classify_loss)


def infinite_iter(data_loader: DataLoader) -> Iterator:
    while True:
        for batch in data_loader:
            yield batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_prefix')
    parser.add_argument('gpu_id')
    parser.add_argument('-ctc', '--ctc_dataset_path', default=None)

    # -svs: resume training
    parser.add_argument('-svs', '--svs_model_path', default=None)
    
    
    args = parser.parse_args()

    dataset_prefix = args.dataset_prefix
    gpu_id = args.gpu_id

    svs_model_path = args.svs_model_path
    ctc_dataset_path = args.ctc_dataset_path

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

    train_dataset, valid_dataset = load_and_process_dataset(dataset_prefix, phoneme_id_list, have_multi_phone)
    train_ctc_dataset = load_and_process_ctc_dataset(ctc_dataset_path, phoneme_id_list, have_multi_phone)

    #args
    learning_rate = hp.learning_rate
    vali_step = hp.vali_step
    total_step = hp.total_step

    torch.multiprocessing.set_sharing_strategy('file_system')

    print ("Use GPU #", gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


    device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')

    # Define model
    model = define_models(device, len(train_ctc_dataset.phoneme_list))
    print("Defined Optimizer and Loss Function.", time.time())
    # Load checkpoint if exists
    model = load_models(model, svs_model_path, device)

    if not os.path.exists(hp.checkpoint_path):
        os.mkdir(hp.checkpoint_path)

    if not os.path.exists(hp.logger_path):
        os.mkdir(hp.logger_path)


    optimizer = torch.optim.AdamW([
                                    {'params': model.target_feature_extractor.parameters()},
                                    {'params': model.score_feature_extractor.parameters()},
                                    {'params': model.audio_content_encoder.parameters()},
                                    {'params': model.pitch_contour_encoder.parameters()},
                                    {'params': model.energy_predictor.parameters()},
                                    {'params': model.cross_attn1.parameters()},
                                    {'params': model.cross_attn2.parameters()},
                                    {'params': model.cross_attn3.parameters()},
                                    {'params': model.ppg_to_latent_feature_share1.parameters()},
                                    {'params': model.ppg_to_latent_feature_share2.parameters()},
                                    {'params': model.decoder.parameters()},
                                ],
                                  lr=learning_rate,
                                  weight_decay=1e-9,
                                  betas=(0.9, 0.999),
                                  eps=1e-9)

    # Get Training Loader
    training_loader = DataLoader(train_dataset,
                                  batch_size=hp.batch_expand_size * hp.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn_note,
                                  num_workers=0)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=hp.batch_expand_size * hp.batch_size,
                              shuffle=False,
                              collate_fn=collate_fn_note,
                              num_workers=0)


    unlabeled_train_loader = DataLoader(
        dataset=train_ctc_dataset,
        batch_size=hp.batch_size,
        shuffle=True,
        collate_fn=ctc_dataset_collate,
        num_workers=0,
    )

    unlabeled_train_iter = infinite_iter(unlabeled_train_loader)

    # Define Some Information
    Start = time.perf_counter()

    writer = SummaryWriter(hp.logger_path)
    best_valid = None
    # Training
    model.train()

    total_split_loss = [0.0 for i in range(7)]

    warmup_steps = hp.n_warm_up_step
    scheduler = get_cosine_schedule_with_warmup(
                optimizer, warmup_steps, total_step
            )

    score_encoder_params = model.score_feature_extractor.parameters()
    other_params = []
    other_params.extend(model.target_feature_extractor.parameters())
    other_params.extend(model.pitch_contour_encoder.parameters())
    other_params.extend(model.energy_predictor.parameters())
    other_params.extend(model.cross_attn1.parameters())
    other_params.extend(model.cross_attn2.parameters())
    other_params.extend(model.cross_attn3.parameters())
    other_params.extend(model.ppg_to_latent_feature_share1.parameters())
    other_params.extend(model.ppg_to_latent_feature_share2.parameters())
    other_params.extend(model.decoder.parameters())

    print ("Start training,", time.time())
    for epoch in range(hp.epochs):
        # read [4, 8, ......] batches
        optimizer.zero_grad()

        current_step = epoch * len(training_loader) + 1

        cur_dropout_rate_for_target_encoder = 0.25 * float(current_step) / float(total_step)
        print ("Current step", current_step, "/", total_step, "set dropout rate for target encoder to", cur_dropout_rate_for_target_encoder)

        model.target_feature_extractor.mask_frame1.p = cur_dropout_rate_for_target_encoder
        model.target_feature_extractor.mask_frame2.p = cur_dropout_rate_for_target_encoder
        model.target_feature_extractor.mask_frame3.p = cur_dropout_rate_for_target_encoder

        for i, batchs in tqdm(enumerate(training_loader)):
            if epoch != 0 or i != 0:
                scheduler.step()

            cur_lr = scheduler.get_last_lr()
            current_step = i + epoch * len(training_loader) + 1
            train_dataset.total_step = current_step

            if current_step > total_step:
                break

            for j, db in enumerate(batchs):
                start_time = time.perf_counter()

                # Get Data
                db = to_device(db, device)
                db['input']['ref_mel_global'] = db['input']['ref_mel'].clone()
                db['input']['ref_mel'] = db['gt']['mel']

                # Forward
                model_output, source_encoding_svs = model(db['input'], ret_ref=False
                    , groundtruth=db["gt"], gt_forcing=True, forcing_ratio=0.75, only_score=False)

                # Supervised loss
                mel_loss, align_loss, pitch_loss, energy_loss, note_dur_prediction_loss = get_supervised_loss(model_output, db['gt'], db['input'])
                t_l = (mel_loss) + (align_loss + pitch_loss + energy_loss + note_dur_prediction_loss / 10.0)
                t_l.backward()

                total_split_loss[0] += align_loss.item() / len(batchs)
                total_split_loss[1] += pitch_loss.item() / len(batchs)
                total_split_loss[2] += mel_loss.item() / len(batchs)
                total_split_loss[3] += energy_loss.item() / len(batchs)
                total_split_loss[4] += note_dur_prediction_loss.item() / len(batchs)
                
                (reconstruct_loss, ctc_classify_loss) = unlabeled_ctc_dataset_forward(model, unlabeled_train_iter
                                                                                        , device, method='append', extra_ref_num=4)
                
                total_unlabeled_loss = ((reconstruct_loss) * 1.0) / 1.0 + ctc_classify_loss * 1.0
                total_unlabeled_loss.backward()

                total_split_loss[5] += reconstruct_loss.item() / len(batchs)
                total_split_loss[6] += ctc_classify_loss.item() / len(batchs)
                
                nn.utils.clip_grad_norm_(
                    score_encoder_params, hp.grad_clip_thresh)
                nn.utils.clip_grad_norm_(
                    other_params, hp.grad_clip_thresh)

                optimizer.step()
                optimizer.zero_grad()

                if j == 0 and i % 50 == 0:
                    print ('{:.4f} {:.4f} {:.4f} {:.4f}'.format(reconstruct_loss.item(), mel_loss.item(), ctc_classify_loss.item()
                                , align_loss.item()))

                if j == 0 and i % 400 == 0:
                    print ("cur lr:", cur_lr[0])

            if current_step % vali_step == 0:
                Now = time.perf_counter()
                # validation
                model.eval()

                total_valid_loss = [0.0 for k in range(7)]
                
                with torch.no_grad():
                    v_cnt = 0
                    v_mel_mpop = 0.0
                    for i_v, vbatchs in enumerate(valid_loader):
                        for j_v, vb in enumerate(vbatchs):

                            vb = to_device(vb, device)
                            ref_mel_gt_mpop = vb["gt"]["mel"]
                            gt_pitch_mpop = vb["gt"]["pitch"][:,:,0].unsqueeze(1)
                            mel_for_ref_encoder = vb["input"]["ref_mel"]
                            ref_energy_mpop = vb["gt"]["energy"].unsqueeze(2)
                            vb['input']['ref_mel_global'] = vb['input']['ref_mel'].clone()

                            v_cnt += 1

                            model_output, source_encoding_svs = model(vb['input'], ret_ref=False, only_score=False)
                            mel_loss, align_loss, pitch_loss, energy_loss, note_dur_prediction_loss = get_supervised_loss(model_output, vb['gt'], vb['input'])
                            # t_l = align_loss.item() + pitch_loss.item() + mel_loss.item() + energy_loss.item()
                            
                            total_valid_loss[0] += align_loss.item()
                            total_valid_loss[1] += pitch_loss.item()
                            total_valid_loss[2] += mel_loss.item()
                            total_valid_loss[3] += energy_loss.item()
                            total_valid_loss[4] += note_dur_prediction_loss.item()


                vb = None
                model.train()

                writer.add_scalars('Loss/align', {'train': total_split_loss[0] / vali_step,
                                            'valid': total_valid_loss[0] / v_cnt}, current_step)

                writer.add_scalars('Loss/pitch', {'train': total_split_loss[1] / vali_step,
                                            'valid': total_valid_loss[1] / v_cnt}, current_step)

                writer.add_scalars('Loss/melspec', {'train': total_split_loss[2] / vali_step,
                                            'valid': total_valid_loss[2] / v_cnt}, current_step)

                writer.add_scalars('Loss/energy', {'train': total_split_loss[3] / vali_step,
                                            'valid': total_valid_loss[3] / v_cnt}, current_step)

                writer.add_scalars('Loss/Note dur (time lag)', {'train': total_split_loss[4] / vali_step,
                                            'valid': total_valid_loss[4] / v_cnt}, current_step)

                writer.add_scalars('Loss/Weakly data melspec', {'train': total_split_loss[5] / vali_step,}, current_step)

                writer.add_scalars('Loss/CTC loss', {'train': total_split_loss[6] / vali_step,}, current_step)

                writer.add_scalar('optimizer/learning_rate', cur_lr[0], current_step)

                str1 = "Epoch [{}/{}], Step [{}/{}]:".format(epoch+1, hp.epochs, current_step, total_step)
                str2 = "Training melspec loss: {:.4f}, weakly melspec loss: {:.4f};".format(total_split_loss[2] / vali_step, total_split_loss[5] / vali_step)
                str3 = "Validation melspec loss: {:.4f};".format(total_valid_loss[2] / v_cnt)
                
                print ("\n" + str1)
                print (str2)
                print (str3)

                total_split_loss = [0.0 for i in range(7)]

                save_dict = model.state_dict()
                target_model_path = os.path.join(hp.checkpoint_path, str(current_step) + '.pth.tar')
                torch.save(save_dict, target_model_path)

                print("save model at step %d ..." % current_step)