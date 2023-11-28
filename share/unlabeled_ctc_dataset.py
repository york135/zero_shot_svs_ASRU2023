import os,re, pickle
import numpy as np
import sys
import librosa

import time

import torch
from torch.utils.data import Dataset

from tqdm import tqdm

from utils import get_pitch, compute_energy, compute_mel, pad_log_2D_tensor
import h5py

# MIR-ST500 !!!
class LyricsDataset(Dataset):
    def __init__(self):
        self.singer_name = []
        self.melspec = []
        self.pitch_contour = []
        self.energy_feature = []
        self.lyrics = []

        self.audio_paths = []
        self.h5_data_paths = []

    def add_weakly_data(self, audio_dir, lyrics_dir, output_h5_dir):
        with torch.no_grad():
            for cur_dir in os.listdir(audio_dir):
                singer_name = cur_dir
                cur_singer_dir = os.path.join(audio_dir, cur_dir)

                cur_lyrics_list = []

                self.audio_paths.append([])
                self.lyrics.append([])
                self.h5_data_paths.append([])

                print (cur_singer_dir)

                for audio_name in tqdm(os.listdir(cur_singer_dir)):
                    audio_path = os.path.join(audio_dir, cur_dir, audio_name)
                    lyrics_path = os.path.join(lyrics_dir, cur_dir, audio_name[:-4] + ".lab")

                    if not os.path.exists(lyrics_path):
                        continue

                    with open(lyrics_path) as f:
                        lyrics = f.readlines()[0].split()

                    voc, sr = librosa.core.load(audio_path, sr=24000, mono=True)
                    # Drop audio clips that are longer than 20s (it will probably cause cuda OOM).
                    if len(voc) > 480000:
                        print ("Drop", audio_path, len(voc))
                        continue

                    voc = librosa.util.normalize(voc) * 0.9
                    voc_mel, _ = compute_mel(voc)

                    energy_curve, abs_energy_curve = compute_energy(voc)
                    max_abs_energy = max(abs_energy_curve)

                    pitch_model_input = librosa.resample(voc, 24000, 16000)
                    pitch_output = get_pitch(pitch_model_input)

                    output_path = os.path.join(output_h5_dir, singer_name + '_' + audio_name[:-4] + '.h5')

                    self.audio_paths[-1].append(audio_path)
                    self.lyrics[-1].append(lyrics)
                    self.h5_data_paths[-1].append(output_path)

                    with h5py.File(output_path, "w") as f:
                        grp = f.create_group("singer_name")
                        grp2 = f.create_group("melspec")
                        grp3 = f.create_group("pitch_contour")
                        grp4 = f.create_group("energy_feature")

                        grp.create_dataset('0', data=singer_name)
                        grp2.create_dataset('0', data=voc_mel)
                        grp3.create_dataset('0', data=pitch_output)
                        grp4.create_dataset('0', data=energy_curve)

                # print (len(cur_ref_list))
                self.singer_name.append(singer_name)

    def get_all_features_from_h5(self, idx, cur_ref_clip_idx):
        dataset_path = self.h5_data_paths[idx][cur_ref_clip_idx]
        cur_file_data = h5py.File(dataset_path, 'r')

        # melspec, pitch_contour, energy_feature
        melspec = cur_file_data["melspec"]['0'][()]
        pitch_contour = cur_file_data["pitch_contour"]['0'][()]
        energy_feature = cur_file_data["energy_feature"]['0'][()]

        return (melspec, pitch_contour, energy_feature)

    def get_melspec(self, idx, cur_ref_clip_idx):
        dataset_path = self.h5_data_paths[idx][cur_ref_clip_idx]
        cur_file_data = h5py.File(dataset_path, 'r')
        return cur_file_data["melspec"]['0'][()]

    def get_random_ref_clips(self, idx, number_of_clips_to_concate=5):
        ref_clip_idx = torch.randint(low=0, high=len(self.audio_paths[idx]), size=(number_of_clips_to_concate,))
        ref_data_target = [self.get_melspec(idx, ref_clip_idx[i]) for i in range(number_of_clips_to_concate)]
        cur_mel_for_ref_encoder = torch.cat([torch.tensor(ref_data_target[i]) for i in range(len(ref_data_target))], dim=0)
        return cur_mel_for_ref_encoder


    def sample_ctc_target_audio(self, idx, melspec, method, extra_ref_num):
        melspec = torch.tensor(melspec)
        if method == 'same' or (method == 'append' and extra_ref_num == 0):
            return melspec, melspec

        elif method == 'append':
            ref_data_target = self.get_random_ref_clips(idx, extra_ref_num)
            mel_for_ref_encoder = torch.cat([ref_data_target, melspec], dim=0)
            return melspec, mel_for_ref_encoder

        elif method == 'different':
            ref_data_target = ref_dataset.get_random_ref_clips(ref_dataset_id[k], extra_ref_num)
            mel_for_ref_encoder = ref_data_target
            return melspec, mel_for_ref_encoder
        else:
            print ("Wrong target sampling method......")

    def enroll_phoneme_list(self, phoneme_list, have_multi_phone):
        # In ver1 (ASRU ver.), I assign two labels for Mandarin finals that contain multiple phonemes (e.g. 'uan').
        # Later on, I found that this does not improve performance, so I removed it.
        # have_multi_phone defines those multi-phoneme finals
        # To not activate this setting, simply passing have_multi_phone=[]
        self.phoneme_list = phoneme_list
        self.have_multi_phone = have_multi_phone
        return

    def lyrics2token(self, lyrics):
        lyrics_token = []
        for i in range(len(lyrics)):
            if lyrics[i] == '<EOS>':
                continue

            cur_lyrics_index = self.phoneme_list.index(lyrics[i])
            lyrics_token.append(cur_lyrics_index)

            if self.have_multi_phone[cur_lyrics_index] == True:
                lyrics_token.append(cur_lyrics_index + 1)

        return lyrics_token

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        # Randomly select training sample for singer[idx]
        ref_clip_idx = int(torch.randint(low=0, high=len(self.h5_data_paths[idx]), size=(1,)))

        lyrics = self.lyrics[idx][ref_clip_idx]
        lyrics_token = self.lyrics2token(lyrics)

        melspec, pitch_contour, energy_feature = self.get_all_features_from_h5(idx, ref_clip_idx)
        
        ref_mel, ref_global_mel = self.sample_ctc_target_audio(idx, melspec, method='append', extra_ref_num=4)

        return {'mel': melspec,
                'ref_mel': ref_mel.numpy(),
                'ref_global_mel': ref_global_mel.numpy(),
                'ref_clip_idx': ref_clip_idx,
                'singer_idx': idx,
                'lyrics_token': lyrics_token,
                'energy': energy_feature,
                'pitch': pitch_contour,
                }


def ctc_dataset_collate(batch):
    data_size = len(batch)

    melspec = [batch[ind]['mel'] for ind in range(len(batch))]
    melspec = pad_log_2D_tensor(melspec, pad_value=-10.0)

    ref_mel = [batch[ind]['ref_mel'] for ind in range(len(batch))]
    ref_mel = pad_log_2D_tensor(ref_mel, pad_value=-10.0)

    ref_global_mel = [batch[ind]['ref_global_mel'] for ind in range(len(batch))]
    ref_global_mel = pad_log_2D_tensor(ref_global_mel, pad_value=-10.0)

    energy = [batch[ind]['energy'] for ind in range(len(batch))]
    energy = pad_log_2D_tensor(energy, pad_value=-10.0)

    pitch = [batch[ind]['pitch'] for ind in range(len(batch))]
    pitch = pad_log_2D_tensor(pitch, pad_value=0)

    lyrics_token = [batch[ind]['lyrics_token'] for ind in range(len(batch))]
    singer_idx = [batch[ind]['singer_idx'] for ind in range(len(batch))]
    ref_clip_idx = [batch[ind]['ref_clip_idx'] for ind in range(len(batch))]

    new_batch = {'mel': melspec,
                'ref_mel': ref_mel,
                'ref_global_mel': ref_global_mel,
                'ref_clip_idx': ref_clip_idx,
                'singer_idx': singer_idx,
                'lyrics_token': lyrics_token,
                'energy': energy,
                'pitch': pitch,
                }

    return new_batch

if __name__ == "__main__":
    dataset_dir = sys.argv[1]
    lyrics_dir = sys.argv[2]
    output_pkl_path = sys.argv[3]
    output_h5_dir = sys.argv[4]

    if not os.path.exists(output_h5_dir):
        os.mkdir(output_h5_dir)

    cur_dataset = LyricsDataset()
    cur_dataset.add_weakly_data(dataset_dir, lyrics_dir, output_h5_dir)

    with open(output_pkl_path, 'wb') as f:
        pickle.dump(cur_dataset, f)

    print (len(cur_dataset))
    print (cur_dataset[0][0])