import os, sys, time, pickle
# sys.path.append(os.path.join(os.path.dirname(__file__), '../share'))
import numpy as np
import torch
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

import librosa
import torchcrepe

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


def get_pitch(vocal, device='cuda'):
    # Sample rate should be 16,00 0Hz
    audio = torch.tensor(np.copy(vocal))[None]
    audio = audio.to(device)

    # 50 Hz frame rate
    hop_length = 80
    fmin = 50 # C2 = 65.406 Hz
    fmax = 1000 # B5 = 987.77 Hz
    model = "full"

    with torch.no_grad():
        pitch = torchcrepe.predict(audio, 16000, hop_length, fmin, fmax, model, batch_size=512, device=device).detach().cpu().numpy()
        pitch_output = np.array([[i*0.005, librosa.hz_to_midi(pitch[0][i])] for i in range(pitch.shape[1])])
    return pitch_output

def compute_energy(y, fft_size=1024, hop_length=120):
    energy = librosa.feature.rms(y=y, frame_length=fft_size, hop_length=hop_length, center=True, pad_mode='reflect')
    log_energy = np.log10(energy + 1e-10)
    # print (energy_feature.shape)
    return log_energy.T, energy.T

def compute_mel(y):
    sample_rate = 24000
    fft_size = 1024
    hop_length = 120

    num_mels = 80
    fmin = 0
    fmax = sample_rate / 2

    eps = 1e-10
    log_base = 10.0

    x_stft = librosa.stft(
        y,
        n_fft=fft_size,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
    )
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel = np.maximum(eps, np.dot(spc, mel_basis.T))
    # Get log spec

    if log_base is None:
        return np.log(mel), spc
    elif log_base == 10.0:
        return np.log10(mel), spc
    elif log_base == 2.0:
        return np.log2(mel), spc
    else:
        raise ValueError(f"{log_base} is not supported.")


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_mask_from_lengths(lengths, device='cuda', max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len).to(device))
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_1D_tensor(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = F.pad(torch.tensor(x), (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded

def pad_2D_tensor(inputs, maxlen=None):

    def pad(x, max_len):
        if x.shape[0] > max_len:
            raise ValueError("not max_len")

        s = x.shape[1]
        x_padded = F.pad(torch.tensor(x), (0, 0, 0, max_len-x.shape[0]))
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.shape[0] for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output

def pad_log_2D_tensor(inputs, maxlen=None, pad_value=0):

    def pad(x, max_len, pad_value):
        if x.shape[0] > max_len:
            raise ValueError("not max_len")

        s = x.shape[1]
        x_padded = F.pad(torch.tensor(x), (0, 0, 0, max_len-x.shape[0]), value=pad_value)
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen, pad_value) for x in inputs])
    else:
        max_len = max(x.shape[0] for x in inputs)
        output = torch.stack([pad(x, max_len, pad_value) for x in inputs])

    return output

def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        out_list = list()
        max_len = mel_max_length
        for i, batch in enumerate(input_ele):
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
            out_list.append(one_batch_padded)
        out_padded = torch.stack(out_list)
        return out_padded
    else:
        out_list = list()
        max_len = max([input_ele[i].size(0)for i in range(len(input_ele))])

        for i, batch in enumerate(input_ele):
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
            out_list.append(one_batch_padded)
        out_padded = torch.stack(out_list)
        return out_padded



def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)


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
    padding_mask = seq_k.eq(PAD)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat

def compute_distance(distance_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                dur = float(duration_predictor_output[i][j])
                # distance to onset
                distance_mat[i][count+k][0] = k
                distance_mat[i][count+k][1] = distance_mat[i][count+k][0] / dur

                # distance to offset
                distance_mat[i][count+k][2] = dur - 1 - k
                distance_mat[i][count+k][3] = distance_mat[i][count+k][2] / dur

            count = count + duration_predictor_output[i][j]
    return distance_mat

def to_device(db, device):
    ret = {}
    phoneme_seq = db["input"]["input_feature"][:,:,0]
    phoneme_tone = db["input"]["input_feature"][:,:,1]
    phoneme_score_pitch = db["input"]["input_feature"][:,:,2]
    word_duration = db["input"]["input_feature"][:,:,3]
    is_multi_phoneme = db["input"]["input_feature"][:,:,5]
    # is_begin_initial_final = db["input"]["input_feature"][:,:,6]

    # Input of the model
    ret["input"] = {
                    "phoneme_seq": phoneme_seq.long().to(device),
                    "phoneme_tone": phoneme_tone.long().to(device),
                    "word_duration": word_duration.float().to(device),
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
                    'note_dur': db["input"]["note_dur"].long().to(device),
                    'score_note_dur': db["input"]["score_note_dur"].long().to(device),
                    }

    # Groundtruth (optional)
    if "gt" in db.keys():
        ret["gt"] = {
                    'mel': db["gt"]["mel"].float().to(device),
                    'energy': db["gt"]["energy"].float().to(device),
                    'pitch': db["gt"]["pitch"].float().to(device),
                    'dur': db["gt"]["dur"].float().to(device),
                    'note_dur': db["input"]["note_dur"].float().to(device),
                    'gt_alignment': db["gt"]["gt_alignment"],
                    'waveforms': db["gt"]["waveforms"].float().to(device),
                    'target_singer': db["gt"]["target_singer"]  # does not need to put target singer to gpu
                    }

    return ret

def compute_phoneme_change(phoneme_labels, note_labels):
    # 10 frames
    group_duration = 0.05
    cur_phoneme_start = 0
    cur_phoneme_end = 0
    phoneme_change_groups = 0
    for i in range(len(phoneme_labels)):
        if phoneme_labels[i] == 0:
            continue
        cur_phoneme_start = cur_phoneme_end
        cur_phoneme_end = cur_phoneme_start + phoneme_labels[i]
        if int(round(cur_phoneme_start / group_duration)) != int(round(cur_phoneme_end / group_duration)):
            phoneme_change_groups = phoneme_change_groups + 1

    frame_groups = int(round(cur_phoneme_end / group_duration))

    return frame_groups, phoneme_change_groups


def load_and_process_dataset(dataset_prefix, phoneme_list, have_multi_phone, test_only=False):
    from dataset import MpopDataset
    # Load datasets from pkl files
    datasets_path = dataset_prefix
    print ('Start loading dataset,', time.time(), 'from', datasets_path)
    gt_train_set = []
    input_train_set = []
    note_data_train_set = []
    waveforms_train_set = []

    gt_test_set = []
    input_test_set = []
    note_data_test_set = []
    waveforms_test_set = []

    lookup_table_train = {}
    lookup_table_test = {}

    singer_ids = ['f1', 'f2', 'm1', 'm2']
    for singer_id in singer_ids:
        lookup_table_train[singer_id] = []
        lookup_table_test[singer_id] = []

    for singer_id in singer_ids:
        cur_path = datasets_path + '_' + singer_id + '.pkl'
        with open(cur_path, 'rb') as f:
            input_feature, note_data, gt_labels, waveforms = pickle.load(f)

        cur_singer_length = len(input_feature)
        train_part_length = int(cur_singer_length * 0.95)

        for i in range(train_part_length):
            gt_train_set.append(gt_labels[i])
            input_train_set.append(input_feature[i])
            note_data_train_set.append(note_data[i])
            waveforms_train_set.append(waveforms[i])
            lookup_table_train[singer_id].append(len(gt_train_set) - 1)
        
        # test set
        for i in range(train_part_length, cur_singer_length):
            gt_test_set.append(gt_labels[i])
            input_test_set.append(input_feature[i])
            note_data_test_set.append(note_data[i])
            waveforms_test_set.append(waveforms[i])
            lookup_table_test[singer_id].append(len(gt_test_set) - 1)

    print (phoneme_list)
    print (have_multi_phone)

    print ("Length of training dataset:", len(gt_train_set))
    print ("Length of testing dataset:", len(gt_test_set)) 
    print ("Split dataset completed,", time.time())


    train_dataset = MpopDataset(gt_train_set, input_train_set, note_data_train_set, lookup_table_train, waveforms_train_set
                , phoneme_list, have_multi_phone, is_test=False)
    valid_dataset = MpopDataset(gt_test_set, input_test_set, note_data_test_set, lookup_table_test, waveforms_test_set
                , phoneme_list, have_multi_phone, is_test=True)
    return train_dataset, valid_dataset