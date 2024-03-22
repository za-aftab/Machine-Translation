#! /bin/bash
# University of Zurich
# Department of Computational Linguistics
# MT Exercise 2: Data, Pre- and Postprocessing

# Authors: Zainab Aftab & Kristina Horn

scripts=`dirname "$0"`
base=$scripts/..

data=$base/data
configs=$base/configs

translations=$base/translations

mkdir -p $translations

src=de
trg=en

# cloned from https://github.com/bricksdont/moses-scripts
MOSES=$base/tools/moses-scripts/scripts

num_threads=6
device=5

model_name=transformer_iwslt14_deen_bpe

### Translation
CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python -m joeynmt translate $configs/$model_name.yaml < $data/test.raw.$src > $translations/test.full.$model_name.$trg

## Depending on your Postprocessing prefrence you can uncomment the corresponding lines.

# Postprocessing Step Full
# cat $translations/test.full.$model_name.$trg | sacrebleu $data/test.raw.$trg

# Postprocessing Step Full but lowercased
# cat $translations/test.full.$model_name.$trg | sacrebleu -lc $data/test.raw.$trg

# Postprocessing Step Tokenized with MosesTokenizer instead
# # Postprocessing
# cat $translations/test.full.$model_name.$trg | $MOSES/tokenizer/tokenizer.perl -l $trg > $translations/test.retokenized.$model_name.$trg

# # BLEU Comparison
# cat $translations/test.retokenized.$model_name.$trg | sacrebleu --tokenize none $data/test.raw.$trg


# Postprocessing Step Tokenize hyp and ref differently
## Postprocessing
cat $translations/test.full.$model_name.$trg | $MOSES/tokenizer/tokenizer.perl -l $trg > $translations/test.retokenized.$model_name.$trg
cat $translations/test.full.$model_name.$trg | $MOSES/tokenizer/tokenizer_PTB.perl -l $trg > $translations/test.retokenized_raw.$model_name.$trg

## BLEU Comparison
cat $translations/test.retokenized.$model_name.$trg | sacrebleu --tokenize none $translations/test.retokenized_raw.$model_name.$trg