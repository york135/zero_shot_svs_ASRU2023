import sys
import os
import time
import argparse
import torch
import numpy as np
import torchcrepe
import librosa
import json
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

def get_pitch(wav_path, device):

    audio, sr = librosa.core.load(wav_path, sr=24000, mono=True)
    audio = librosa.util.normalize(audio)

    hop_length = 120
    hop_second = 120.0 / 24000.0

    audio = torch.tensor(np.copy(audio))[None]
    audio = audio.to(device)

    fmin = 50 # C2 = 65.406 Hz
    fmax = 1000 # B5 = 987.77 Hz

    model = "full"

    pitch, confidence = torchcrepe.predict(audio, 24000, hop_length, fmin, fmax, model, batch_size=512, device=device, return_periodicity=True)
    pitch = pitch.cpu().numpy()

    win_length = 3
    confidence = torchcrepe.filter.median(confidence, win_length)
    confidence = torchcrepe.threshold.Silence(-60.)(confidence,
                                                 audio,
                                                 24000,
                                                 hop_length)

    confidence = confidence.cpu().numpy()

    # print (pitch)
    pitch_output = np.array([[i*hop_second, librosa.hz_to_midi(pitch[0][i]), confidence[0][i]] for i in range(pitch.shape[1])])
    return pitch_output


def run_crepe(input_path, output_path, device):
    pitch_result = get_pitch(input_path, device)
    pitch_result = pitch_result.tolist()

    # Write results to target file
    with open(output_path, 'w') as f:
        output_string = json.dumps(pitch_result)
        f.write(output_string)

    return pitch_result


if __name__ == "__main__":
    dataset_path = sys.argv[1]
    print (time.time())
    device = sys.argv[2]
    singer_names = ['f1', 'f2', 'm1', 'm2']

    for singer_name in singer_names:
        print (singer_name)
        
        cur_path = os.path.join(dataset_path, singer_name)
        audio_dir = os.path.join(cur_path, 'audio')

        if not os.path.isdir(os.path.join(cur_path, 'pitch')):
            os.mkdir(os.path.join(cur_path, 'pitch'))

        all_song_name = []
        for audio in tqdm(os.listdir(audio_dir)):
            if os.path.isfile(os.path.join(audio_dir, audio)):
                input_path = os.path.join(audio_dir, audio)
                output_path = os.path.join(cur_path, 'pitch', audio[:-4] + '.json')
                # print (input_path, output_path, device)
                pitch_result = run_crepe(input_path, output_path, device)
                # print (len(pitch_result))
                # print (pitch_result[-100:])

    print (time.time())
