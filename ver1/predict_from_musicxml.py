import os, sys, re, time
import textgrid
import xml.etree.ElementTree as ET
import dragonmapper.transcriptions
import dragonmapper.hanzi
import music21
import pickle
import librosa

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '../share'))
import utils
from utils import process_text, pad_1D, pad_2D, pad_2D_tensor, pad_log_2D_tensor, pad_1D_tensor, compute_mel
from model import PWGVocoder, AutoSVS
import soundfile as sf

from tqdm import tqdm


# copy from FastSpeech2
def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

# Need this lexicon file (can obtain from FastSpeech2) to tranlate syllables into phoneme
lexicon = read_lexicon("../share/process_mpop/pinyin-lexicon-r.txt")
frame_per_second = 200.0
sample_rate = 24000
batch_expand_size = 1
batch_size = 1

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

class TestDataset(Dataset):
    def __init__(self, input_data, note_data, ref, target_length, phoneme_list, have_multi_phone):
        # print (phoneme_list)
        # print (have_multi_phone)

        phoneme_order_in_note = []
        for i in range(len(input_data)):
            # print (input_data[i])
            input_data[i][0], cur_phoneme_orders = reformulate_phoneme_seq(input_data[i][0], phoneme_list, have_multi_phone)
            input_data[i][0] = np.array(input_data[i][0])
            phoneme_order_in_note.append(np.array(cur_phoneme_orders))

        # process note data, convert duration to frame number.
        for i in range(len(note_data)):
            last_frame = 0
            last_time = 0
            for j in range(len(note_data[i])):
                last_time = last_time + note_data[i][j][1]
                cur_frame = int(round(last_time * frame_per_second))
                note_data[i][j][1] = cur_frame - last_frame
                last_frame = cur_frame

        self.input_data = input_data
        self.note_data = note_data
        self.ref = ref
        self.target_length = target_length
        self.phoneme_list = phoneme_list
        self.phoneme_order_in_note = phoneme_order_in_note
        
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return {
                'input_feature': self.input_data[idx][0], #[phoneme->one-hot, note duration, note_pitch]
                'phoneme_order': self.phoneme_order_in_note[idx],
                'ref_mel': self.ref,
                'target_length': self.target_length[idx],
                'note_data': self.note_data[idx]
                }

def process_xml(xml_path):
    xml_data = music21.converter.parse(xml_path)
    vocal_notes = []

    # perform a simple time shift first
    first_note_onset = None

    for part in xml_data.parts:
        instrument = part.getInstrument().instrumentName
        mapped_list = xml_data.flat.secondsMap
        for event in mapped_list:
            if getattr(event['element'], 'isNote', None) and event['element'].isNote and event['element'].lyric is not None:
                if len(vocal_notes) == 0:
                    first_note_onset = event['offsetSeconds']
                vocal_notes.append([event['offsetSeconds'], event['endTimeSeconds'], event['element'].pitch.midi, event['element'].lyric])

            elif getattr(event['element'], 'tie', None) is not None:
                # print (event['element'].tie)
                vocal_notes[-1][1] = event['endTimeSeconds']

    return vocal_notes

def segment_notes(vocal_notes):
    segments = []
    segment_offsets = []
    target_length = []

    last_offset = 0.0
    cur_segment = []
    cur_segment_offset = 0.0
    for i in range(len(vocal_notes)):
        if len(cur_segment) > 0 and vocal_notes[i][0] > last_offset + 0.01:
            segments.append(cur_segment)
            segment_offsets.append(cur_segment_offset)
            cur_segment = [vocal_notes[i],]
            cur_segment_offset = vocal_notes[i][0]
        elif len(cur_segment) == 0:
            cur_segment = [vocal_notes[i],]
            cur_segment_offset = vocal_notes[i][0]
        else:
            cur_segment.append(vocal_notes[i])
        last_offset = vocal_notes[i][1]

    if len(cur_segment) > 0:
        segments.append(cur_segment)
        segment_offsets.append(cur_segment_offset)

    for i in range(len(segments)):
        for j in range(len(segments[i])):
            segments[i][j][0] = segments[i][j][0] - segment_offsets[i]
            segments[i][j][1] = segments[i][j][1] - segment_offsets[i]

        target_length.append(int(segments[i][-1][1] * frame_per_second) + 1)

    return segments, segment_offsets, target_length

def generate_feature(vocal_segments):
    note_data = []
    input_feature = []
    for i in range(len(vocal_segments)):
        cur_note_data = []
        cur_input_feature = []

        # Add phoneme-level data and <sep> token
        cur_input_feature.append(['sep', 0, 0.0, 0.0])
        last_phoneme = None
        
        for j in range(len(vocal_segments[i])):
            cur_note_data.append([vocal_segments[i][j][2], vocal_segments[i][j][1] - vocal_segments[i][j][0]])

            if str(vocal_segments[i][j][3]) == '-':
                cur_phoneme_seq = [last_phoneme, ]

            else:
                if str(vocal_segments[i][j][3]) == '妳':
                    vocal_segments[i][j][3] = '你'

                if str(vocal_segments[i][j][3]) == '淆':
                    vocal_segments[i][j][3] = '搖'

                if str(vocal_segments[i][j][3]) == '波':
                    vocal_segments[i][j][3] = '坡'

                pinyin_syllable = dragonmapper.hanzi.to_pinyin(str(vocal_segments[i][j][3]), accented=False)
                pinyin_syllable = pinyin_syllable.replace('ü', 'v')
                if pinyin_syllable == 'o1':
                    pinyin_syllable = 'ou1'

                if pinyin_syllable == 'jing5':
                    pinyin_syllable = 'jing1'

                cur_phoneme_seq = lexicon[pinyin_syllable]

            for k in range(len(cur_phoneme_seq)):
                tone = cur_phoneme_seq[k][-1]
                if tone == "1" or tone == "2" or tone == "3" or tone == "4" or tone == "5":
                    cur_input_feature.append([cur_phoneme_seq[k][:-1], int(tone), vocal_segments[i][j][2], vocal_segments[i][j][1] - vocal_segments[i][j][0]])
                else:
                    cur_input_feature.append([cur_phoneme_seq[k], 0, vocal_segments[i][j][2], vocal_segments[i][j][1] - vocal_segments[i][j][0]])

            last_phoneme = str(cur_phoneme_seq[-1])

            cur_input_feature.append(['sep', 0, 0.0, 0.0])

        input_feature.append([cur_input_feature, 'test'])
        note_data.append(cur_note_data)
    return note_data, input_feature


  
def reprocess_tensor_note(batch, cut_list):
    # masking padded frames
    # for note-level input
    input_features = [batch[ind]['input_feature'] for ind in cut_list]
    phoneme_order = [batch[ind]['phoneme_order'] for ind in cut_list]

    length_word = np.array([])
    for word in input_features:
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
    target_lengths = [batch[ind]['target_length'] for ind in cut_list]
    length_spec = np.array(list())
    for target_length in target_lengths:
        length_spec = np.append(length_spec, target_length)

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

    note_data = [np.array(batch[ind]['note_data']) for ind in cut_list]
    note_data = pad_2D_tensor(note_data)

    note_pitch = note_data[:,:,0]
    note_dur = note_data[:,:,1]

    out = {"input": 
                    {"input_feature": input_features,
                    "phoneme_order": phoneme_order,
                    "word_pos": word_pos,
                    "spec_pos": spec_pos,
                    "spec_max_len": max_spec_len,
                    "mel_pos": mel_pos,
                    "max_mel_len": max_mel_len,
                    'ref_mel': ref_mel,
                    'note_pitch': note_pitch,
                    'score_note_dur': note_dur}
            }

    return out


def test_collate_fn_note(data):
    data_size = len(data)
    real_batchsize = batch_expand_size

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
    return output

def test_to_device(db, device):
    ret = {}
    phoneme_seq = db["input"]["input_feature"][:,:,0]
    phoneme_tone = db["input"]["input_feature"][:,:,1]
    phoneme_score_pitch = db["input"]["input_feature"][:,:,2]
    is_multi_phoneme = db["input"]["input_feature"][:,:,4]

    # Input of the model
    ret["input"] = {
                    "phoneme_seq": phoneme_seq.long().to(device),
                    "phoneme_tone": phoneme_tone.long().to(device),
                    "is_multi_phoneme": is_multi_phoneme.to(device),
                    "phoneme_score_pitch": phoneme_score_pitch.long().to(device),
                    "phoneme_order": db["input"]["phoneme_order"].long().to(device),
                    "word_pos": db["input"]["word_pos"].long().to(device),
                    "spec_pos": db["input"]["spec_pos"].long().to(device),
                    "spec_max_len": db["input"]["spec_max_len"],
                    "mel_pos": db["input"]["mel_pos"].long().to(device),
                    "max_mel_len": db["input"]["max_mel_len"],
                    'ref_mel': db["input"]["ref_mel"].float().to(device),
                    'note_pitch': db["input"]["note_pitch"].long().to(device),
                    'score_note_dur': db["input"]["score_note_dur"].long().to(device),
                    'note_dur': db["input"]["score_note_dur"].long().to(device),
                    }

    return ret 

def predict_song(model_path, test_loader, output_path, segment_offsets):
    device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')
    # device = 'cpu'
    # Define model
    print("Use FastSpeech,", time.time())
    model = AutoSVS(device=device).to(device)
    vocoder = PWGVocoder(device=device, normalize_path='../share/pwg_pretrained/train_nodev_all_1010_pwg.v1.no_limit/stats.npy'
        , vocoder_path='../share/pwg_pretrained/train_nodev_all_1010_pwg.v1.no_limit/checkpoint-400000steps.pkl').to(device)

    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of SVS Parameters:', num_param)

    num_param_vocoder = utils.get_param_num(vocoder)
    print('Number of vocoder Parameters:', num_param_vocoder)

    checkpoint = torch.load(model_path, map_location=device)
        
    model.load_state_dict(checkpoint, strict=True)

    print("\n---Model Restored---\n", model_path)
    print ("Start inference,", time.time())

    model.eval()

    pred_list = []

    with torch.no_grad():
        v_cnt = 0
        v_loss = 0.0
        target_encoding = []

        for i_v, vbatchs in enumerate(tqdm(test_loader)):
            for j_v, vb in enumerate(vbatchs):
                vb = test_to_device(vb, device)

                vb['input']['ref_mel_global'] = vb['input']['ref_mel']

                if len(target_encoding) == 0:
                    model_output, _ = model(vb['input'], ret_ref=True, only_score=True)
                    target_encoding = model_output['target_encoding']
                else:
                    model_output, _ = model(vb['input'], ret_ref=True, only_score=True, pre_computed_target=True, target_encoding=target_encoding)
                
                # print (model_output['mel'].shape)

                waveform_output, _ = vocoder(model_output['mel'], output_all=True)
                pred_list.append(waveform_output[0].cpu().numpy())


    output_wav = np.zeros(int(segment_offsets[-1] * 24000) + len(pred_list[-1]) + 12000)
    # Combine prediction of each segment
    for i in range(len(pred_list)):
        start = int(round(segment_offsets[i] * 24000))
        end = start + len(pred_list[i])
        output_wav[start:end] = pred_list[i]

    sf.write(
        output_path,
        output_wav,
        24000,
        "PCM_16",
    )

if __name__ == "__main__":
    model_path = sys.argv[1]
    gpu_id = sys.argv[2]
    xml_path = sys.argv[3]
    ref_path = sys.argv[4]
    output_path = sys.argv[5]

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # model_name example: 'best_checkpoint.pth.tar'

    vocal_notes = process_xml(xml_path)
    vocal_segments, segment_offsets, target_length = segment_notes(vocal_notes)
    note_data, input_feature = generate_feature(vocal_segments)

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

    # Get reference audio and its mel spec
    ref_y, sr = librosa.core.load(ref_path, sr=sample_rate, mono=True)
    ref_y = librosa.util.normalize(ref_y) * 0.9
    ref_mel_feature, ref_spc = compute_mel(ref_y)

    test_dataset = TestDataset(input_feature, note_data, ref_mel_feature, target_length, phoneme_id_list, have_multi_phone)
    test_loader = DataLoader(test_dataset,
                              batch_size=batch_expand_size * batch_size,
                              shuffle=False,
                              collate_fn=test_collate_fn_note,
                              num_workers=0)
    

    predict_song(model_path, test_loader, output_path, segment_offsets)