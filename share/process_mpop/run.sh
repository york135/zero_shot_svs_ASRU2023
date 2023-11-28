#!/usr/bin/env bash

# Copyright 2018 open-speech songmeixu (songmeixu@outlook.com)

set -e

stage=1

cd ./speech-aligner/egs/cn_phn
. ./path.sh
. ./utils/parse_options.sh


# generate alignments
if [ $stage -le 1 ]; then
  speech-aligner --config=conf/align.conf ../../../wav.scp ../../../text ../../../out.ali || exit 1;
fi

exit 0;
