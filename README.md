# zero_shot_svs_ASRU2023

This is the official implementation of the paper "Zero-shot singing voice synthesis from musical score," which will be presented at ASRU 2023 conference.

I release two versions of the source code. The first one (`ver1`) is the source code required to reproduce our experiments in the paper. The second one (`ver2`) is a better implementation I made after submitting the manuscript to ASRU 2023. The results of `ver2` sounds better than `ver1` (my subjective feeling). But to maintain the reproducibility, I decide to release both versions. The shared files are put under `utils` folder.

## Term of usage

**Source code**: Free to use!

**Pretrained models**: For **ALL** the pretrained models that I release, you can **ONLY** use them for **academic use**. Commercial usage is strictly prohibited. This is due to the term of usage of the dataset that the pretrained models were trained on. 

If you use my source code to train models with different datasets, you can decide the term of usage of them without any restriction. But please respect the term of usage of those datasets.

**Disclaimers.** I do not hold responsible for any damage caused by the misuse (illegal, criminal, or others) of this repo by the users.

Last but not least, zero-shot SVS models may be misused to clone other singers' voice in an illegal way (though currently my model is still not that good, in the near future, I will update a more powerful one!), please be careful.

## Credits

- About model architecture: [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2), [yistLin/FragmentVC](https://github.com/yistLin/FragmentVC)

- About evaluation: [clovaai/voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer) (I used the ResNetSE34L checkpoint as the backbone singer verification model)

- About vocoder: [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN) (I used the Parallel WaveGAN setting)

## Install requirements

```
pip install -r requirements.txt
```

## Dataset requirements

To reproduce the experiments, you need to obtain the **MPOP600 dataset** (https://ieeexplore.ieee.org/document/9306461) and the **OpenSinger dataset** ([GitHub - Multi-Singer/Multi-Singer.github.io](https://github.com/Multi-Singer/Multi-Singer.github.io)). To obtain the former one, you may have to contact the authors of that paper.

### Musdb-V test set

As for the Musdb-V dataset that we used for evaluation, the dataset was proposed in [USVG](https://github.com/bryan051003/USVG) (I was the second author of that paper). Unfortunately, the first author (who conducted the experiments) did not open-source these materials. Therefore, I upload the Musdb-V dataset [here](https://drive.google.com/drive/folders/119OFiSUuFE3wZP3T6-xl-Xq2EghnMm48?usp=sharing). 

To create this dataset, we:

- Identify the singer identity of the vocal tracks in the Musdb-18 dataset.

- Group all the clips sung by the same singer. (Note that even within the same song, there may be multiple singers. We carefully separate these parts. We also discard phrases where multiple singers sing simultaneously)

- Remove silence parts.

Note that I only merged the clips from the same singer in the test set, as I only used the merged version of Musdb-V test set for the experiments.

## Preprocessing

### MPOP600 dataset

See `share/process_mpop/readme.md`.

### Support for the M4Singer dataset

Since the label files of the MPOP600 dataset is a little bit messy, this repo also provide the preprocessing code for the M4Singer dataset ([GitHub - M4Singer/M4Singer](https://github.com/M4Singer/M4Singer)). Based on my subjective feeling, training with M4Singer yields a better model than training with MPOP600. 

However, since M4Singer does not provide the musical score (the `meta.json` file only has note-level and phoneme-level labels), we cannot train the time-lag model (the difference between the actual onset/offset timing and the onset/offset timing in the musical score) with that dataset.

(Under construction)

### OpenSinger dataset (as weakly labeled data)

First, modify the file structure of the dataset as follows:

```
OpenSinger_audio (contain audios, can have arbitrary name)
|----0 (singer name)
|    |----0_斑马，斑马_0.wav
|    |----0_斑马，斑马_1.wav
|    |         ...
|----1
|    |----1_爱转角_0.wav
|    |----1_爱转角_1.wav
|    |         ...
OpenSinger_lyrics (contain lyrics, can have arbitrary name)
|----0 (singer name)
|    |----0_斑马，斑马_0.lab
|    |----0_斑马，斑马_1.lab
|    |         ...
|----1
|    |----1_爱转角_0.lab
|    |----1_爱转角_1.lab
|    |         ...
```

Note that this is **NOT** the original file structure of the OpenSinger dataset. I made some changes to put all the audio clips sung by the same singer under the same directory. By doing so, my program can easily identify the singer of a song by simply refering to the directory name.

Also, OpenSinger assigned repeated singer names (male: 0-27; female: 0-47). To deal with this issue, I added '100' to the female singers' name. For example, female singer '45' would be referred to as '145'.

**Dataset partition**: Singer 25, 26, 27, 145, 146, 147 (3 males, 3 females) are left out for testing. Clips from all other singers (70 in total) are used for training.



Then, run:

```
cd share/
python unlabeled_ctc_dataset.py [dataset_dir] [lyrics_dir] \ 
                            [output_pkl_path] [output_h5_dir]
```

where `dataset_dir` specifies the directory to OpenSinger's audio files; `lyrics_dir` specifies the directory to OpenSinger's lyrics files.

The code generates one pickle file to `output_pkl_path` and $n$ h5 files to the `output_h5_dir` directory, where $n$ is the number of total audio clips. The pickle file contains the information of singer name, lyrics, and the paths to the h5 files.

## Training

There are multiple versions in this repo. The commands to run train/evaluation code are the same, but you have to go to the directory you like. Assume you want to use `ver1`:

```
cd ver1/
python train_whole_model.py [labeled_dataset_prefix] [gpu_id] \
                          -ctc [ctc_dataset_path]
```

where `labeled_dataset_prefix` specifies the prefix of the labeled data (see `share/process_mpop/readme.md`); `gpu_id` specifies the GPU id; `ctc_dataset_path` specifies the pickle file for the weakly labeled data (i.e., the `output_pkl_path` in `unlabeled_ctc_dataset.py`).

You may want to modify `hparams.py` to change the directory of model checkpoints, tensorboard logger files, or try different settings.

## Evaluation (SVAR, cosine similarity)

Assume you want to use `ver1`:

```
cd ver1/
python predict_from_musicxml_for_asv.py [gpu_id]
```

Other paths are written in `predict_from_musicxml_for_asv.py`. You may want to change it. We need the musicxml files in the MPOP600 dataset and the audios in the OpenSinger test set for evaluation. The audio clips sung by the same singer and have the same song name in OpenSinger test set should be merged beforehand (e.g. `25_回到过去_0.wav`, `25_回到过去_2.wav`, `25_回到过去_3.wav`, ......, should be merged to form a single clip `25_回到过去.wav`).

`share/opensinger_test_merge.json` specifies the {musical score (musicxml): reference audio (wav)} pairs used for evaluation. These pairs were sampled randomly and were used to provide objective evaluation results in our paper. 

By default, the musical scores and reference audios should be put at `../mpop600_test_score`  and `../opensinger_test_merged`, respectively. If this is not the case, modify and run `share/change_asv_path.py` to change the two directories.

## Pretrained models

Here are the pretrained models:

| Model URL                                                                                  | Setting | Paper            | Training dataset                               | Term of use       |
|:------------------------------------------------------------------------------------------:|:-------:|:----------------:|:----------------------------------------------:|:-----------------:|
| [Here](https://drive.google.com/file/d/1MhBzlM6ffCavXP53Ub3tyCPfxZK6MLLP/view?usp=sharing) | ver1    | Here (ASRU 2023) | MPOP600 (labeled), OpenSinger (weakly labeled) | Only academic use |
