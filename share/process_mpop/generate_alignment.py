import os
import librosa
import soundfile
from tqdm import tqdm

def run_speech_aligner(words, pinyins, audio_path):

    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    # print (pinyins)
    cur_pinyin_idx = 0
    for i in tqdm(range(len(words))):
        if words[i][3] != 'bre' and words[i][3] != 'sil':
            try:
                start = int(round(16000.0 * words[i][0]))
                end = int(round(16000.0 * words[i][1]))
                cur_clip = audio[start:end]
                
                soundfile.write('temp.wav', cur_clip, samplerate=16000, subtype='PCM_16')

                with open('text', 'w') as f:
                    f.write('temp ' + pinyins[cur_pinyin_idx])

                os.system('bash run.sh >>alignment_log 2>&1')

                with open('out.ali', 'r') as f:
                    contents = f.read()
                    # print (contents)
                    phoneme_time = contents.split('\n')[1:]

                # print (phoneme_time)

                first_non_slience = 0.0
                if phoneme_time[0].split()[2] == 'sil' or phoneme_time[0].split()[2] == 'bre':
                    word_dur = words[i][1] - words[i][0]
                    first_non_slience = float(phoneme_time[1].split()[0])
                    aligned_word_dur = float(phoneme_time[2].split()[1]) - float(phoneme_time[1].split()[0])

                else:
                    word_dur = words[i][1] - words[i][0]
                    first_non_slience = float(phoneme_time[0].split()[0])
                    aligned_word_dur = float(phoneme_time[1].split()[1]) - float(phoneme_time[0].split()[0])

                # print (word_dur, first_non_slience, aligned_word_dur)

                aligned_ratio = word_dur / aligned_word_dur

                for j in range(len(phoneme_time)):
                    segmented = phoneme_time[j].split()
                    if len(segmented) < 3:
                        continue
                    onset = (float(segmented[0]) - first_non_slience) * aligned_ratio + words[i][0]
                    offset = (float(segmented[1]) - first_non_slience) * aligned_ratio + words[i][0]
                    phoneme_name = segmented[2]

                    if phoneme_name != 'sil':
                        words[i][5].append([onset, offset, phoneme_name])

                # print (words[i][5])
                words[i][5][0][0] = words[i][0]
                words[i][5][-1][1] = words[i][1]

            except:
                last_idx = max(0, i - 1)
                next_idx = min(i + 1, len(words) - 1)

                start = int(round(16000.0 * words[last_idx][0]))
                end = int(round(16000.0 * words[next_idx][1]))
                cur_clip = audio[start:end]
                soundfile.write('temp.wav', cur_clip, samplerate=16000, subtype='PCM_16')

                print (start, end, words[last_idx:next_idx])

                # print (cur_clip)
                

                with open('text', 'w') as f:
                    f.write('temp ' + pinyins[max(0, cur_pinyin_idx - 1)] + ' ' + pinyins[cur_pinyin_idx] + ' ' + pinyins[min(len(pinyins) - 1, cur_pinyin_idx + 1)])

                os.system('bash run.sh >>alignment_log 2>&1')

                with open('out.ali', 'r') as f:
                    contents = f.read()
                    # print (contents)
                    phoneme_time = contents.split('\n')

                    if words[last_idx][3] == 'sil' or  words[last_idx][3] == 'bre':
                        phoneme_time = phoneme_time[2:4]
                    elif phoneme_time[1].split()[2] == 'sil' or phoneme_time[1].split()[2] == 'bre':
                        phoneme_time = phoneme_time[4:6]
                    else:
                        phoneme_time = phoneme_time[3:5]

                word_dur = words[i][1] - words[i][0]
                aligned_word_dur = float(phoneme_time[-1].split()[1]) - float(phoneme_time[0].split()[0])
                first_non_slience = float(phoneme_time[0].split()[0])

                print (word_dur, aligned_word_dur)

                aligned_ratio = word_dur / aligned_word_dur

                for j in range(len(phoneme_time)):
                    segmented = phoneme_time[j].split()
                    if len(segmented) < 3:
                        continue
                    onset = (float(segmented[0]) - first_non_slience) * aligned_ratio + words[i][0]
                    offset = (float(segmented[1]) - first_non_slience) * aligned_ratio + words[i][0]
                    phoneme_name = segmented[2]

                    if phoneme_name != 'sil':
                        words[i][5].append([onset, offset, phoneme_name])

                print (words[i][5])
                words[i][5][0][0] = words[i][0]
                words[i][5][-1][1] = words[i][1]

            cur_pinyin_idx = cur_pinyin_idx + 1

    return words