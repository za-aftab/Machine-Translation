#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools
samples=$base/samples

mkdir -p $samples

num_threads=4
device=""

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/decleration \
        --words 300 \
        --checkpoint $models/model_00.pt \
        --outf $samples/sample_00
)

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/decleration \
        --words 300 \
        --checkpoint $models/model_01.pt \
        --outf $samples/sample_01
)

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/decleration \
        --words 300 \
        --checkpoint $models/model_05.pt \
        --outf $samples/sample_05
)

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/decleration \
        --words 300 \
        --checkpoint $models/model_07.pt \
        --outf $samples/sample_07
)

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/decleration \
        --words 300 \
        --checkpoint $models/model_09.pt \
        --outf $samples/sample_09
)
