import argparse
import logging
import os, sys
import time

import numpy as np
import soundfile as sf
import torch
import yaml

from tqdm import tqdm

from parallel_wavegan.datasets import MelDataset
from parallel_wavegan.datasets import MelSCPDataset
from parallel_wavegan.utils import load_model
from parallel_wavegan.utils import read_hdf5

import librosa
from sklearn.preprocessing import StandardScaler

sample_rate = 24000
fft_size = 1024
hop_length = 120

hop_time_duration = hop_length / sample_rate

num_mels = 80
fmin = 0
fmax = sample_rate / 2

def compute_mel(y):
    
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
    mel_basis = librosa.filters.mel(sample_rate, fft_size, num_mels, fmin, fmax)
    mel = np.maximum(eps, np.dot(spc, mel_basis.T))

    # Get log spec
    # spc = librosa.amplitude_to_db(spc)

    # print (mel.shape)
    if log_base is None:
        return np.log(mel), spc
    elif log_base == 10.0:
        return np.log10(mel), spc
    elif log_base == 2.0:
        return np.log2(mel), spc
    else:
        raise ValueError(f"{log_base} is not supported.")


if __name__ == "__main__":
    print (time.time())
    # device = torch.device('cpu')
    source_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_path = sys.argv[3]
    stats_path = sys.argv[4]

    print ("source dir", source_dir)
    print ("target dir", output_dir)
    print ("Model path", model_path)
    print ("stat path", stats_path)

    # source_path = sys.argv[1]
    # output_path = sys.argv[2]
    device = sys.argv[5]

    # print ("source path", source_path)
    # print ("target path", output_path)

    if torch.cuda.is_available():
        device = torch.device(device)
    else:
        device = torch.device("cpu")

    print ("use", device)
    # model = load_model('./checkpoint-400000steps.pkl')
    # model.remove_weight_norm()
    # model = model.eval().to(device)

    # # Construct scaler
    # stats_path = './stats.h5'
    # scaler = StandardScaler()

    # scaler.mean_ = read_hdf5(stats_path, "mean")
    # scaler.scale_ = read_hdf5(stats_path, "scale")

    # print (scaler.mean_)
    # print (scaler.scale_)

    # scaler.n_features_in_ = scaler.mean_.shape[0]

    model = load_model(model_path)

    model.remove_weight_norm()
    model = model.eval().to(device)

    scaler = StandardScaler()
    data = np.load(stats_path)

    scaler.mean_ = data[0]
    scaler.scale_ = data[1]

    print (scaler.mean_)
    print (scaler.scale_)

    scaler.n_features_in_ = scaler.mean_.shape[0]

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # for data_name in os.listdir(source_dir):
    #     audio_path = os.path.join(source_dir, data_name)
    #     output_path = os.path.join(output_dir, data_name)

    #     print (audio_path, output_path)
    #     # audio_path = source_path
    #     y, sr = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    #     y = librosa.util.normalize(y) * 0.9
    #     mel_feature, spc = compute_mel(y)

    #     mel_feature = scaler.transform(mel_feature)

    #     result_audio = []

    #     with torch.no_grad():
    #         for i in tqdm(range(0, len(mel_feature), 2000)):
    #             start = i
    #             end = min(i + 2000, len(mel_feature))
    #             c = torch.tensor(mel_feature[start:end], dtype=torch.float).to(device)
    #             # print (mel_feature.shape, c.shape, mel_feature)
    #             new_y = model.inference(c).view(-1)

    #             if len(result_audio) < 2:
    #                 result_audio = new_y.cpu()
    #             else:
    #                 result_audio = torch.cat((result_audio, new_y.cpu()))
    #             # print (new_y, y.shape, result_audio.shape)

    #     # save as PCM 16 bit wav file
    #     sf.write(
    #         output_path,
    #         result_audio.numpy(),
    #         24000,
    #         "PCM_16",
    #     )
    for singer_name in os.listdir(source_dir):
        cur_source_dir = os.path.join(source_dir, singer_name)
        cur_output_dir = os.path.join(output_dir, singer_name)

        if not os.path.isdir(cur_output_dir):
            os.mkdir(cur_output_dir)

        for data_name in os.listdir(cur_source_dir):
            audio_path = os.path.join(cur_source_dir, data_name)
            output_path = os.path.join(cur_output_dir, data_name)

            print (audio_path, output_path)
            # audio_path = source_path
            y, sr = librosa.core.load(audio_path, sr=sample_rate, mono=True)
            y = librosa.util.normalize(y) * 0.9
            mel_feature, spc = compute_mel(y)

            mel_feature = scaler.transform(mel_feature)

            result_audio = []

            with torch.no_grad():
                for i in tqdm(range(0, len(mel_feature), 2000)):
                    start = i
                    end = min(i + 2000, len(mel_feature))
                    c = torch.tensor(mel_feature[start:end], dtype=torch.float).to(device)
                    # print (mel_feature.shape, c.shape, mel_feature)
                    new_y = model.inference(c).view(-1)

                    if len(result_audio) < 2:
                        result_audio = new_y.cpu()
                    else:
                        result_audio = torch.cat((result_audio, new_y.cpu()))
                    # print (new_y, y.shape, result_audio.shape)

            # save as PCM 16 bit wav file
            sf.write(
                output_path,
                result_audio.numpy(),
                24000,
                "PCM_16",
            )