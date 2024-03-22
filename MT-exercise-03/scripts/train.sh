#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools

mkdir -p $models

num_threads=4
device=""

SECONDS=0

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/decleration \
        --epochs 40 \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout 0.0 --tied \
        --save $models/model_00.pt \
        --save_ppl out_00.txt
)

echo "time taken:"
echo "$SECONDS seconds"


(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/decleration \
        --epochs 40 \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout 0.1 --tied \
        --save $models/model_01.pt \
        --save_ppl out_01.txt
)

echo "time taken:"
echo "$SECONDS seconds"

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/decleration \
        --epochs 40 \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout 0.5 --tied \
        --save $models/model_05.pt \
        --save_ppl out_05.txt
)

echo "time taken:"
echo "$SECONDS seconds"


(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/decleration \
        --epochs 40 \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout 0.7 --tied \
        --save $models/model_07.pt \
        --save_ppl out_07.txt
)

echo "time taken:"
echo "$SECONDS seconds"

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/decleration \
        --epochs 40 \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout 0.9 --tied \
        --save $models/model_09.pt \
        --save_ppl out_09.txt
)

echo "time taken:"
echo "$SECONDS seconds"

