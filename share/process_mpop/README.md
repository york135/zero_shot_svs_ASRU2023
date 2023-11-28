# Preprocessing MPOP600

## Overview

We need to run two steps to preprocess MPOP600:

1) Obtaining phoneme-level and frame-level features, including phoneme alignment (using `speech-aligner`) and pitch (F0) contour (using `torchcrepe`). 

2) Combining these features with other audio features (log-Mel spectrogram, energy, etc) to form the training data.

The first step is pretty tiresome and ad-hoc. It is only used to deal with MPOP600's annotation. Its annotation contains various labels (musicxml, zhuyin, textgrid) that have to be aligned together, but lacks the crucial phoneme duration (alignment) labels.

．If you have difficulty reproducing the first step, you can directly e-mail me. I will send you the link to the processed labels. Note that these labels **do not** contain audio (I am not supposed to redistribute it). You still have to obtain the audio files from the owners of the MPOP600 dataset.

Here, I provide the code to run these steps.

## Install speech-aligner and torchcrepe

- Download speech-aligner from [https://github.com/open-speech/speech-aligner](https://github.com/open-speech/speech-aligner) (`git clone https://github.com/open-speech/speech-aligner.git`). Put it under this directory.

- Follow the instruction of speech-aligner to compile it. Make sure that `./speech-aligner/bin/speech-aligner` exists. During preprocessing, `generate_alignment.py` will call `run.sh`, which will attempt call this binary file to perform alignment.

- (Optional) Change the `retry-beam` parameter in `speech-aligner/egs/cn_phn/conf/align.conf` to `200` to ensure that `speech-aligner` always produces an alignment. 

- Install `dragonmapper`, `music21`, `TextGrid`, and `torchcrepe` via `pip`. These packages are required to generate phoneme labels and pitch contour.

## MPOP600 dataset structure

The structure of the files should be as follows:

```
MPOP600 (dataset parent directory)
|----f1
|    |----audio
|    |    |----f1_006.wav
|    |    |----f1_007.wav
|    |         ...
|    |----注音
|    |    |----f1_006_bopo.trs
|    |    |----f1_007_bopo.trs
|    |         ...
|    |----label
|    |    |----f1_006.TextGrid
|    |    |----f1_007.TextGrid
|    |         ...
|    |----score
|    |    |----f1_006.musicxml
|    |    |----f1_007.musicxml
|    |         ...
```

Basically, when I received the MPOP600 dataset from the owners in August 2021, this is how it was organized.

Note that we leave the first 5 songs for each singer as the test set. All files related to the first 5 songs should be put under another parent directory.

## Obtain phoneme-level labels

```
python generate_phoneme_label.py [dataset_directory]
```

The phoneme-level labels will be stored as pickle files under `phoneme_level` directory. 

## Obtain pitch contour

```
python run_crepe.py [dataset_directory] [device]
```

The pitch contour will be stored under `pitch` directory.

## Step 2: Combining all labels

```
python generate_dataset.py mpop600 [dataset_directory] [output_prefix]
```

This will generate one pickle file for each singer. For MPOP600, the number of singers is 4. `output_prefix` specifies the prefix of these pickle files. For example, passing `../../train` will create `../../train_m1.pkl`, `../../train_m2.pkl`, `../../train_f1.pkl`, and `../../train_f2.pkl`.
