import pickle, os, sys
import json
import librosa
import numpy as np
from tqdm import tqdm
import music21

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils import compute_energy, compute_mel

def check_is_valid(word_segments):
    valid_segments = []
    for i in range(len(word_segments)):
        valid_flag = 1
        for j in range(len(word_segments[i])):
            # For every phone label, check if it is valid.
            if len(word_segments[i][j][4]) > len(word_segments[i][j][5]):
                valid_flag = 0
                break

            for k in range(len(word_segments[i][j][5])):
                if word_segments[i][j][5][k][0] >= word_segments[i][j][5][k][1]:
                    valid_flag = 0
                    break
            if valid_flag == 0:
                break

        if valid_flag == 1:
            valid_segments.append(word_segments[i])
        else:
            print ("invalid segment:", word_segments[i])

    return valid_segments


def time_to_frame_MPOP600(y, gt_labels, input_feature, waveforms, word_segments, mel_feature
    , energy_feature, pitch_data, singer_name, use_sep=True):
    
    # input: notes [duration, note pitch, phoneme seq]
    for i in range(len(word_segments)):
        cur_segments_input = []
        # initial/final-level w/ alignment
        for j in range(len(word_segments[i])):
            if use_sep:
                cur_segments_input.append(['sep', 0, 0.0, 0.0, 0.0, 0, 0])

            for k in range(len(word_segments[i][j][4])):
                # initial/final name (no tone), tone, pitch, note duration, score note duration, is_slur, phoneme pos in syllable
                # is_slur is always 0 for MPOP600
                tone = word_segments[i][j][4][k][-1]
                if tone == "1" or tone == "2" or tone == "3" or tone == "4" or tone == "5":
                    cur_segments_input.append([word_segments[i][j][4][k][:-1], int(tone), word_segments[i][j][2]
                        , word_segments[i][j][1] - word_segments[i][j][0], word_segments[i][j][6], 0, k+1])
                else:
                    cur_segments_input.append([word_segments[i][j][4][k], 0, word_segments[i][j][2]
                        , word_segments[i][j][1] - word_segments[i][j][0], word_segments[i][j][6], 0, k+1])

        if use_sep:
            cur_segments_input.append(['sep', 0, 0.0, 0.0, 0.0, 0, 0])

        input_feature.append([cur_segments_input, singer_name])
        # frame-level
        # generate gt mel, pitch, phoneme duration, energy
        hop_time_duration = 120.0 / 24000.0
        start = int(round(word_segments[i][0][0] / hop_time_duration))
        end = int(round(word_segments[i][-1][1] / hop_time_duration))

        gt_mel = mel_feature[start:end]
        gt_energy = energy_feature[start:end]
        gt_pitch = pitch_data[start:end, 1:3]

        wave_start = int(round(word_segments[i][0][0] * sample_rate))
        wave_end = int(round(word_segments[i][-1][1] * sample_rate))

        waveforms.append(y[wave_start:wave_end])

        gt_phoneme_duration = []
        # for every note
        for j in range(len(word_segments[i])):
            # if only one phoneme (e.g. "ai")
            if len(word_segments[i][j][4]) == 1:
                gt_phoneme_duration.append(0.0)
                gt_phoneme_duration.append(word_segments[i][j][1] - word_segments[i][j][0])
            else:
                if len(word_segments[i][j][4]) != len(word_segments[i][j][5]):
                    print ("error", word_segments[i][j])
                gt_phoneme_duration.append(0.0)
                for k in range(len(word_segments[i][j][4])):
                    gt_phoneme_duration.append(word_segments[i][j][5][k][1] - word_segments[i][j][5][k][0])

        gt_phoneme_duration.append(0.0)
        gt_labels.append([gt_mel, gt_energy, gt_pitch, gt_phoneme_duration])

    return gt_labels, input_feature, waveforms

def segment_words(words, vocal_notes):
    word_segments = []
    cur_segments = []

    cur_note_id = 0
    try:
        for i in range(len(words)):
            if (words[i][3] == 'sil' or words[i][3] == 'bre'):
                if cur_segments != []:
                    word_segments.append(cur_segments)
                    cur_segments = []
            else:
                cur_segments.append(words[i])
                cur_segments[-1].append(vocal_notes[cur_note_id][1] - vocal_notes[cur_note_id][0])
                cur_note_id = cur_note_id + 1
    except:
        print (cur_segments)
        print (word_segments)

        for i in range(len(word_segments)):
            for j in range(len(word_segments[i])):
                print (word_segments[i][j], word_segments[i][j][1] - word_segments[i][j][0])

    word_segments = check_is_valid(word_segments)
    return word_segments

def collect_note_feature(word_segments, input_note_feature):
    for i in range(len(word_segments)):
        cur_segments_input = []
        # phoneme-level
        for j in range(len(word_segments[i])):
            # Note pitch, note duration, score note duration
            cur_segments_input.append([word_segments[i][j][2], word_segments[i][j][1] - word_segments[i][j][0], word_segments[i][j][6]])

        # print (cur_segments_input)
        input_note_feature.append(cur_segments_input)
    return input_note_feature


def read_musicxml(score_path):
    xml_data = music21.converter.parse(score_path)
    vocal_notes = []

    for part in xml_data.parts:
        instrument = part.getInstrument().instrumentName
        mapped_list = xml_data.flat.secondsMap
        for event in mapped_list:
            if getattr(event['element'], 'isNote', None) and event['element'].isNote and event['element'].lyric is not None:
                vocal_notes.append([event['offsetSeconds'], event['endTimeSeconds'], event['element'].pitch.midi, event['element'].lyric])

            elif getattr(event['element'], 'tie', None) is not None and vocal_notes[-1][1] == event['offsetSeconds']:
                vocal_notes[-1][1] = event['endTimeSeconds']

    return vocal_notes

def generate_MPOP600_dataset(dataset_path, singers_list, output_prefix):
    for singer_name in singers_list:
        gt_labels = []
        input_feature = []
        input_note_feature = []
        waveforms = []

        if os.path.isfile(output_prefix + '_' + singer_name + '.pkl'):
            continue

        print (singer_name)

        cur_path = os.path.join(dataset_path, singer_name)
        audio_dir = os.path.join(cur_path, 'audio')

        for audio in tqdm(os.listdir(audio_dir)):
            audio_path = os.path.join(audio_dir, audio)

            if os.path.isfile(os.path.join(audio_dir, audio)):
                y, sr = librosa.core.load(audio_path, sr=sample_rate, mono=True)
                y = librosa.util.normalize(y) * 0.9
                mel_feature, spc = compute_mel(y)
                energy_feature, _ = compute_energy(y)

                pkl_path = os.path.join(cur_path, 'phoneme_level', audio[:-4] + '.pkl')
                with open(pkl_path, 'rb') as f:
                    words = pickle.load(f)
                
                pitch_path = os.path.join(cur_path, 'pitch', audio[:-4] + '.json')
                with open(pitch_path) as f:
                    pitch_data = np.array(json.load(f))

                score_path = os.path.join(cur_path, 'score', audio[:-4] + '.musicxml')
                vocal_notes = read_musicxml(score_path)

                word_segments = segment_words(words, vocal_notes)
                input_note_feature = collect_note_feature(word_segments, input_note_feature)

                gt_labels, input_feature, waveforms = time_to_frame_MPOP600(y, gt_labels, input_feature, waveforms
                                                , word_segments, mel_feature, energy_feature, pitch_data, singer_name)

        with open(output_prefix + '_' + singer_name + '.pkl', 'wb') as f:
            pickle.dump([input_feature, input_note_feature, gt_labels, waveforms], f)

def generate_M4Singer_dataset(dataset_path, output_prefix):
    for singer_name in singers_list:
        gt_labels = []
        input_feature = []
        input_note_feature = []
        waveforms = []

        if os.path.isfile(output_prefix + '_' + singer_name + '.pkl'):
            continue

        print (singer_name)

        cur_path = os.path.join(dataset_path, singer_name)
        audio_dir = os.path.join(cur_path, 'audio')

        for audio in tqdm(os.listdir(audio_dir)):
            audio_path = os.path.join(audio_dir, audio)

            if os.path.isfile(os.path.join(audio_dir, audio)):
                y, sr = librosa.core.load(audio_path, sr=sample_rate, mono=True)
                y = librosa.util.normalize(y) * 0.9
                mel_feature, spc = compute_mel(y)
                energy_feature, _ = compute_energy(y)

                pkl_path = os.path.join(cur_path, 'phoneme_level', audio[:-4] + '.pkl')
                with open(pkl_path, 'rb') as f:
                    words = pickle.load(f)
                
                pitch_path = os.path.join(cur_path, 'pitch', audio[:-4] + '.json')
                with open(pitch_path) as f:
                    pitch_data = np.array(json.load(f))

                score_path = os.path.join(cur_path, 'score', audio[:-4] + '.musicxml')
                vocal_notes = read_musicxml(score_path)

                word_segments = segment_words(words, vocal_notes)
                input_note_feature = collect_note_feature(word_segments, input_note_feature)

                gt_labels, input_feature, waveforms = time_to_frame_MPOP600(y, gt_labels, input_feature, waveforms
                                                , word_segments, mel_feature, energy_feature, pitch_data, singer_name)

        with open(output_prefix + '_' + singer_name + '.pkl', 'wb') as f:
            pickle.dump([input_feature, input_note_feature, gt_labels, waveforms], f)



if __name__ == "__main__":
    dataset_type = sys.argv[1]
    dataset_path = sys.argv[2]
    output_prefix = sys.argv[3]
    # e.g. python preprocess_dataset.py ../../MPOP600/ ../231123
    if dataset_type == 'mpop600':
        singers_list = ['f1', 'f2', 'm1', 'm2']
        generate_MPOP600_dataset(dataset_path, singers_list, output_prefix)
    elif dataset_type == 'm4singer':
        generate_M4Singer_dataset(dataset_path, output_prefix)