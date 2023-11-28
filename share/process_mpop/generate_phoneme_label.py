import os, sys, re
import textgrid
import xml.etree.ElementTree as ET
import dragonmapper.transcriptions
import music21
import generate_alignment
import pickle

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

# Need this lexicon file to tranlate syllables into phoneme
# Borrow from FastSpeech2 (https://github.com/ming024/FastSpeech2/blob/d4e79eb52e8b01d24703b2dfc0385544092958f3/lexicon/pinyin-lexicon-r.txt)
lexicon = read_lexicon("pinyin-lexicon-r.txt")

def handle_bad_textgrid(textgrid_path):
    words = []
    first_word_onset = None

    with open(textgrid_path, 'r') as f:
        contents = f.read()
        tg_data = contents.split('\n')[12:]

    for i in range(0, len(tg_data), 3):

        if i + 2 >= len(tg_data):
            break
        text = tg_data[i+2][1:-1]
        onset = float(tg_data[i])
        offset = float(tg_data[i+1])
        # The "None" element here stands for note pitch, which will be determined from score label later.
        # The "['sil',]" element here stands for pinyin phoneme (default is 'sil'), which will be determined from 注音 labels later.
        words.append([onset, offset, None, text, ['sil',], []])

        if first_word_onset == None and (text != 'sil' and text != 'bre'):
            first_word_onset = onset

    print (words)
    print ('This textgrid', textgrid_path, 'is parsed by my code, not textgrid package')
    return words, first_word_onset


# Process dataset
def process_one_song(dataset_path, singer_name, song_name):
    # word boundary label (with phoneme information)
    print ("start pre-process", singer_name, "'s song", song_name)
    textgrid_path = os.path.join(dataset_path, singer_name, 'label', song_name + '.TextGrid')
    tg = textgrid.TextGrid()
    
    try:
        tg.read(textgrid_path)

        words = []
        first_word_onset = None
        for i in range(len(tg.tiers[0].intervals)):
            text = tg.tiers[0].intervals[i].mark
            onset = tg.tiers[0].intervals[i].minTime
            offset = tg.tiers[0].intervals[i].maxTime
            # The "None" element here stands for note pitch, which will be determined from score label later.
            # The "['sil',]" element here stands for pinyin phoneme (default is 'sil'), which will be determined from 注音 labels later.
            words.append([onset, offset, None, text.strip(), ['sil',], []])

            if first_word_onset == None and (text != 'sil' and text != 'bre'):
                first_word_onset = onset
    except:
        words, first_word_onset = handle_bad_textgrid(textgrid_path)

    # 注音 label
    zhuyin_path = os.path.join(dataset_path, singer_name, '注音', song_name + '_bopo.trs')
    tree = ET.parse(zhuyin_path)
    root = tree.getroot()[0][0][0]
    # word_and_pinyin = []
    pinyins = []
    # print (words)
    for i in range(len(root)):
        word = root[i]
        tails = word.tail[1:-1].split('_')

        # sil or bre or other symbols, should not be translated into pinyin
        if len(tails) < 2:
            continue

        # Convert '注音' (Zhuyin) syllable labels to pinyin syllable labels 
        if tails[1] == 'ㄛ':
            pinyin_syllable = 'ou1'
        elif tails[1] == 'ㄛˊ':
            pinyin_syllable = 'ou2'
        elif tails[1] == 'ㄧㄞˊ':
            pinyin_syllable = 'ya2'
        else:
            pinyin_syllable = dragonmapper.transcriptions.zhuyin_syllable_to_pinyin(tails[1], accented=False)

        pinyin_syllable = pinyin_syllable.replace('ü', 'v')

        # Find alignment between '注音' labels and the 'label' (textgrid) labels
        # Since MPOP600 has multiple labels, I have to use these somewhat ad-hoc methods to determine the actual labels
        best_error = 1000.0
        best_word_id = None
        for j in range(len(words)): 
            dist = abs(float(word.attrib['time']) - words[j][0])
            if dist < best_error and tails[0] == words[j][3]:
                best_error = dist
                best_word_id = j

        pinyins.append(pinyin_syllable)
        words[best_word_id][4] = lexicon[pinyin_syllable]

    audio_path = os.path.join(dataset_path, singer_name, 'audio', song_name + '.wav')

    # Use speech-aligner to find the phoneme-level alignment following MPOP600's paper.
    # The dataset I obtained from MPOP600's owner does not contain phoneme-level alignment, so I have to did it by myself
    words = generate_alignment.run_speech_aligner(words, pinyins, audio_path)

    # vocal notes
    score_path = os.path.join(dataset_path, singer_name, 'score', song_name + '.musicxml')
    xml_data = music21.converter.parse(score_path)

    vocal_notes = []

    # perform a simple time shift first
    first_note_onset = None

    for part in xml_data.parts:
        instrument = part.getInstrument().instrumentName
        # print (instrument)
        mapped_list = xml_data.flat.secondsMap
        for event in mapped_list:
            if getattr(event['element'], 'isNote', None) and event['element'].isNote and event['element'].lyric is not None:
                # print (event['offsetSeconds'], event['endTimeSeconds'], event['element'].pitch.midi, event['element'].lyric)
                if len(vocal_notes) == 0:
                    first_note_onset = event['offsetSeconds']
                vocal_notes.append([event['offsetSeconds'], event['endTimeSeconds'], event['element'].pitch.midi, event['element'].lyric])

            elif getattr(event['element'], 'tie', None) is not None and vocal_notes[-1][1] == event['offsetSeconds']:
                # print (event['element'].tie)
                vocal_notes[-1][1] = event['endTimeSeconds']

    time_shift = first_note_onset - first_word_onset

    for i in range(len(vocal_notes)):
        vocal_notes[i][0] = vocal_notes[i][0] - time_shift
        vocal_notes[i][1] = vocal_notes[i][1] - time_shift

    # Find the alignment between the musicxml labels and the textgrid labels
    # Again, somewhat ad-hoc

    for i in range(len(words)):
        # filter those silence words
        if words[i][3] == 'sil' or words[i][3] == 'bre':
            continue
        best_error = 1000.0
        best_note_id = None
        for j in range(len(vocal_notes)): 
            dist = abs(vocal_notes[j][0] - words[i][0])
            if dist < best_error and vocal_notes[j][3] == words[i][3]:
                best_error = dist
                best_note_id = j
        words[i][2] = vocal_notes[best_note_id][2]

    return words

if __name__ == "__main__":
    dataset_path = sys.argv[1]
    # dataset_path = './'
    singer_names = ['f1', 'f2', 'm1', 'm2']
    for singer_name in singer_names:
        print (singer_name)
        cur_path = os.path.join(dataset_path, singer_name)

        audio_dir = os.path.join(cur_path, 'audio')
        all_song_name = []
        for audio in os.listdir(audio_dir):
            if os.path.isfile(os.path.join(audio_dir, audio)):
                all_song_name.append(audio[:-4])

        if not os.path.isdir(os.path.join(cur_path, 'phoneme_level')):
            os.mkdir(os.path.join(cur_path, 'phoneme_level'))

        for song_name in all_song_name:

            output_path = os.path.join(cur_path, 'phoneme_level', song_name + '.pkl')

            if os.path.isfile(output_path):
                print ("skip", output_path)
                continue
            else:
                print (output_path)

            words = process_one_song(dataset_path, singer_name, song_name)

            with open(output_path, 'wb') as f:
                pickle.dump(words, f, protocol=4)