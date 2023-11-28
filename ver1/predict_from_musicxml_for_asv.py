import os, sys, re, time, json
import xml.etree.ElementTree as ET
import dragonmapper.transcriptions
import dragonmapper.hanzi
import music21
import pickle
import librosa

import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '../share'))
import utils
from utils import process_text, pad_1D, pad_2D, pad_2D_tensor, pad_log_2D_tensor, pad_1D_tensor, compute_mel
from model import PWGVocoder, AutoSVS, CNNASVmodel
import soundfile as sf

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../share/get_speaker_emb'))
from ref_pretrain_encoder import *
import ResNetSE34L


from predict_from_musicxml import *

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

# Need this lexicon file (can obtain from FastSpeech2) to tranlate syllables into phoneme
lexicon = read_lexicon("../share/process_mpop/pinyin-lexicon-r.txt")
frame_per_second = 200.0
sample_rate = 24000
batch_expand_size = 1
batch_size = 1

segment_group_num = 5
def predict_song(model, vocoder, test_loader, segment_offsets, output_prefix=None):
    pred_list = []

    with torch.no_grad():
        v_cnt = 0
        v_loss = 0.0
        cur_wav_id = 0
        result_wav = []
        target_encoding = []

        for i_v, vbatchs in enumerate(test_loader):
            for j_v, vb in enumerate(vbatchs):
                vb = test_to_device(vb, device)
                vb['input']['ref_mel_global'] = vb['input']['ref_mel']
                
                # print (vb['input']['ref_mel'].shape)
                # print (len(target_encoding))
                if len(target_encoding) == 0:
                    model_output, _ = model(vb['input'], ret_ref=True, only_score=True)
                    target_encoding = model_output['target_encoding']
                else:
                    model_output, _ = model(vb['input'], ret_ref=True, only_score=True, pre_computed_target=True, target_encoding=target_encoding)
                # model_output = model(vb['input'], ret_ref=True)
                waveform_output, _ = vocoder(model_output['mel'], output_all=True)
                
                result_wav.append(waveform_output.cpu().numpy()[0])
                
                if len(result_wav) == segment_group_num:
                    output_wav = (np.zeros(int((segment_offsets[(cur_wav_id+1)*segment_group_num-1] - segment_offsets[cur_wav_id*segment_group_num]) * 24000) 
                        + len(result_wav[-1]) + 100))
                    # Combine prediction of each segment
                    for i in range(len(result_wav)):
                        start = int(round((segment_offsets[cur_wav_id*segment_group_num+i] - segment_offsets[cur_wav_id*segment_group_num]) * 24000))
                        end = start + len(result_wav[i])
                        output_wav[start:end] = result_wav[i]

                    cur_clip_output_path = output_prefix + '_' + str(cur_wav_id) + '.wav'
                    output_wav = np.nan_to_num(output_wav)

                    sf.write(
                        cur_clip_output_path,
                        output_wav,
                        24000,
                        "PCM_16",
                    )

                    result_wav = []
                    cur_wav_id = cur_wav_id + 1

    
    if len(result_wav) > 0:
        output_wav = (np.zeros(int((segment_offsets[-1] - segment_offsets[cur_wav_id*segment_group_num]) * 24000) + len(result_wav[-1]) + 100))
        # Combine prediction of each segment
        for i in range(len(result_wav)):
            start = int(round((segment_offsets[cur_wav_id*segment_group_num+i] - segment_offsets[cur_wav_id*segment_group_num]) * 24000))
            end = start + len(result_wav[i])
            output_wav[start:end] = result_wav[i]

        cur_clip_output_path = output_prefix + '_' + str(cur_wav_id) + '.wav'
        output_wav = np.nan_to_num(output_wav)

        sf.write(
            cur_clip_output_path,
            output_wav,
            24000,
            "PCM_16",
        )

        result_wav = []
        cur_wav_id = cur_wav_id + 1

def get_ref(ref_model, y):
    with torch.no_grad():
        y = librosa.resample(y, orig_sr=24000, target_sr=16000)
        y = torch.tensor(y, dtype=torch.float)

        data = y.reshape(-1,y.size()[-1]).to(device)
        data_processed = ref_model.pre_net(data)
        outp, _ = ref_model(data_processed)

    return outp.cpu()


def load_ref_datasets(ref_dataset_path):
    with open(ref_dataset_path, 'rb') as f:
        cur_ref_dataset = pickle.load(f)
    print ("Read reference unsupervised dataset in", ref_dataset_path, "successfully.")
    print (len(cur_ref_dataset))
    return cur_ref_dataset

spk_sim_func = nn.CosineSimilarity(dim=1)

def concate_all_ref_audio_musdb(ref_dir):
    # print (ref_dir)
    ref_audio_list = []
    clip_name_list = os.listdir(ref_dir)

    selected_clip_name_list = []
    for i in range(len(clip_name_list)):

        cur_clip_name = str(i + 1).zfill(2) + '.wav'
        selected_clip_name_list.append(cur_clip_name)


    # for audio_name in tqdm(clip_name_list):
    for audio_name in tqdm(selected_clip_name_list):
        ref_path = os.path.join(ref_dir, audio_name)
        # print (ref_path)
        try:
            ref_audio, _ = librosa.core.load(ref_path, sr=sample_rate, mono=True)
            ref_audio_list.append(np.array(ref_audio))
        except:
            continue

    ref_audio_concate = np.concatenate(ref_audio_list, axis=0)

    return ref_audio_concate

def compute_asv_sim(phoneme_id_list, have_multi_phone, model, vocoder, asv_model, asv_dnn_model
    , device, objective_list, output_dir, loggger_path, threshold, do_synth=True, use_objective_list=False):
    total_avg_accuracy = 0
    total_avg_sim = 0.0

    split_sim = {}
    split_accuracy = {}
    split_results = {}

    # print (objective_list)
    # Male
    for ref_dir, ref_list in objective_list.items():
        print (time.time())
        # ref_dir = os.path.join(male_ref_dataset_dir, ref_singer_name)

        # Collect all speaker embedding from ref dir
        ref_singer_name = ref_dir.split(os.sep)[-1]

        print ("Speaker:", ref_singer_name, ref_dir)

        if use_objective_list == False:
            ref_y = concate_all_ref_audio_musdb(ref_dir)
            ref_y = librosa.util.normalize(ref_y) * 0.9
            ref_mel_feature, ref_spc = compute_mel(ref_y)

            spk_audio = librosa.resample(ref_y, orig_sr=24000, target_sr=16000)

            with torch.no_grad():
                data = torch.tensor(spk_audio, dtype=torch.float).unsqueeze(0)
                outp, _ = asv_model.pre_net(data)
                target_output_feature = asv_model(outp)
                target_output_feature = asv_dnn_model(target_output_feature)
            

        singer_total_clips = 0
        singer_correct = 0
        singer_sim = 0
        singer_sim_list = []
        for musicxml_path in objective_list[ref_dir].keys():

            if use_objective_list == True:
                ref_audio_path = objective_list[ref_dir][musicxml_path]
                ref_y, _ = librosa.core.load(ref_audio_path, sr=sample_rate, mono=True)
                ref_y = librosa.util.normalize(ref_y) * 0.9
                ref_mel_feature, ref_spc = compute_mel(ref_y)

                spk_audio = librosa.resample(ref_y.copy(), orig_sr=24000, target_sr=16000)

                with torch.no_grad():
                    data = torch.tensor(spk_audio, dtype=torch.float).unsqueeze(0)
                    outp, _ = asv_model.pre_net(data)
                    target_output_feature = asv_model(outp)
                    target_output_feature = asv_dnn_model(target_output_feature)

            cur_score = os.path.basename(musicxml_path)
            vocal_notes = process_xml(musicxml_path)
            vocal_segments, segment_offsets, target_length = segment_notes(vocal_notes)
            note_data, input_feature = generate_feature(vocal_segments)

            test_dataset = TestDataset(input_feature, note_data, ref_mel_feature, target_length, phoneme_id_list, have_multi_phone)
            test_loader = DataLoader(test_dataset,
                                      batch_size=batch_expand_size * batch_size,
                                      shuffle=False,
                                      collate_fn=test_collate_fn_note,
                                      num_workers=0)
            
            output_prefix = os.path.join(output_dir, cur_score.split(".")[0] + "_" + ref_singer_name)
            
            if do_synth:
                predict_song(model, vocoder, test_loader, segment_offsets, output_prefix=output_prefix)
            
            total_wav_num = ((len(test_dataset) - 1) // segment_group_num) + 1

            predictions = []

            with torch.no_grad():
                for i in range(total_wav_num):
                    output_path = output_prefix  + '_' + str(i) + '.wav'
                    cur_wav, _ = librosa.core.load(output_path, sr=16000, mono=True)
                    cur_wav = np.nan_to_num(cur_wav)
                    cur_wav = librosa.util.normalize(cur_wav) * 0.9
                    
                    data = torch.tensor(cur_wav, dtype=torch.float).unsqueeze(0)
                    outp, fb_result = asv_model.pre_net(data)
                    # outp = asv_model.pre_net(data)
                    output_feature_asv = asv_model(outp)

                    output_feature = asv_dnn_model(output_feature_asv)

                    if output_feature.isnan().any():
                        print (output_feature_asv, output_feature, target_output_feature, outp, data)
                        print (data.isnan().any())
                        print (fb_result)
                        print (fb_result.isnan().any())

                    cur_sim = spk_sim_func(output_feature, target_output_feature)
                    singer_sim = singer_sim + cur_sim
                    
                    if float(cur_sim) > threshold:                    
                        singer_correct = singer_correct + 1

                    singer_total_clips = singer_total_clips + 1
                    singer_sim_list.append(float(cur_sim))

        print ("Speaker:", ref_singer_name, "avg sim:", singer_sim / singer_total_clips, "total clip:", singer_total_clips
            , "SV Accuracy:", singer_correct / singer_total_clips)

        split_sim[ref_singer_name] = float(singer_sim / singer_total_clips)
        split_accuracy[ref_singer_name] = singer_correct / singer_total_clips
        split_results[ref_singer_name] = list(singer_sim_list)

        total_avg_sim = total_avg_sim + (singer_sim / singer_total_clips)
        total_avg_accuracy = total_avg_accuracy + (singer_correct / singer_total_clips)

    avg_sim = total_avg_sim / len(objective_list)
    avg_accuracy = total_avg_accuracy / len(objective_list)

    print ("Avg result sim", avg_sim, "Avg SV accuracy:", avg_accuracy)

    with open(loggger_path, 'wb') as f:
        pickle.dump([split_sim, split_accuracy, float(total_avg_sim), total_avg_accuracy, split_results], f)


if __name__ == "__main__":

    objective_list_path = '../share/opensinger_test_merge.json'
    objective_model_path = '../share/opensinger_asv_4000step.pth.tar'
    threshold_path = "../share/threshold_dict_opensinger_test_0223.pkl"

    model_path = "model_0311_propose_300000.pth.tar"
    loggger_path = "model_0311_propose_test_opensinger_300000_logger.json"
    output_dir = "model_0311_propose_test_300000"

    use_objective_list = True

    gpu_id = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

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


    # Load model
    device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')
    # Define model
    print("Use FastSpeech,", time.time())
    print ("Use", device)
    model = AutoSVS(device=device, phoneme_num=len(phoneme_id_list)).to(device)
    vocoder = PWGVocoder(device=device, normalize_path='../share/pwg_pretrained/train_nodev_all_1010_pwg.v1.no_limit/stats.npy'
        , vocoder_path='../share/pwg_pretrained/train_nodev_all_1010_pwg.v1.no_limit/checkpoint-400000steps.pkl').to(device)

    print("SVS Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of TTS Parameters:', num_param)


    asv_model = ResNetSE34L.MainModel(nOut=512, log_input=True, initial_model="../share/get_speaker_emb/baseline_lite_ap.model")
    asv_model.eval()

    asv_dnn_model = CNNASVmodel(device='cpu')

    checkpoint = torch.load(objective_model_path, map_location=device)
    asv_dnn_model.load_state_dict(checkpoint, strict=True)
    asv_dnn_model.eval()

    print("\n---ASV Model Restored---\n", objective_model_path, time.time())

    with open(threshold_path, 'rb') as f:
        # thresh_dict = pickle.load(f)
        threshold = pickle.load(f)
        print ("ASV threshold:", threshold)

    with open(objective_list_path) as json_data:
        objective_list = json.load(json_data)

    print ("Total singer number", len(objective_list))

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)

    print("\n---Model Restored---\n", model_path, time.time())
    model.eval()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    compute_asv_sim(phoneme_id_list, have_multi_phone, model, vocoder, asv_model, asv_dnn_model, device
                    , objective_list, output_dir, loggger_path, threshold=threshold, do_synth=True, use_objective_list=use_objective_list)
