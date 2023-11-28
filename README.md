# zero_shot_svs_ASRU2023

This is the official implementation of the paper "Zero-shot singing voice synthesis from musical score," which will be presented at ASRU 2023 conference.

I release two versions of the source code. The first one (`ver1/`) is the source code required to reproduce our experiments in the paper. The second one (`ver2/`) is a better implementation I made after submitting the manuscript to ASRU 2023. The results of `ver2` sounds better than `ver1` (my subjective feeling). But to maintain the reproducibility, I decide to release both versions. The shared files are put under `utils` folder.

CREDIT: Some of the source code is modified from FastSpeech's repo ([GitHub - xcmyz/FastSpeech: The Implementation of FastSpeech based on pytorch.](https://github.com/xcmyz/FastSpeech)).

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

## Dataset requirements

To reproduce the experiments, you need to obtain the **MPOP600 dataset** (https://ieeexplore.ieee.org/document/9306461) and the **OpenSinger dataset** ([GitHub - Multi-Singer/Multi-Singer.github.io](https://github.com/Multi-Singer/Multi-Singer.github.io)). To obtain the former one, you may have to contact the authors of that paper.

### Musdb-V test set

As for the Musdb-V dataset that we used for evaluation, the dataset was proposed in USVG ([GitHub - bryan051003/USVG: A unified model for zero-shot singing voice conversion and synthesis](https://github.com/bryan051003/USVG)), in which I was the second author. Unfortunately, the first author (who conducted the experiments) did not open-source these materials. Therefore, I upload the Musdb-V dataset [here](https://drive.google.com/drive/folders/119OFiSUuFE3wZP3T6-xl-Xq2EghnMm48?usp=sharing). 

To create this dataset, we:

- Identify the singer identity of the vocal tracks in the Musdb-18 dataset.

- Group the clips sung by the same singer. (Note that even within the same song, there may be multiple singers. We carefully separate these parts. We also discard phrases where multiple singers sing simultaneously)

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

Also, OpenSinger assigned repeated singer names (male: 0~27; female: 0~47). To deal with this issue, I added '100' to the female singers' name. For example, female singer '45' would be referred to as '145'.

**Dataset partition**: Singer 25, 26, 27, 145, 146, 147 (3 males, 3 females) are left out for testing. Clips from all other singers (70 in total) are used for training.

## Pretrained models

Here are the pretrained models:

| Model URL                                                                                  | Setting | Paper            | Training dataset                               | Term of use       |
|:------------------------------------------------------------------------------------------:|:-------:|:----------------:|:----------------------------------------------:|:-----------------:|
| [Here](https://drive.google.com/file/d/1MhBzlM6ffCavXP53Ub3tyCPfxZK6MLLP/view?usp=sharing) | ver1    | Here (ASRU 2023) | MPOP600 (labeled), OpenSinger (weakly labeled) | Only academic use |
