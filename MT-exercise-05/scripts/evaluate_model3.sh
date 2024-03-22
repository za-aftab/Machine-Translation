#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data
configs=$base/configs

translations=$base/translations

mkdir -p $translations

src=nl ##
trg=de ##


num_threads=4
device=0

# measure time

SECONDS=0

model_name=transformer_model3 ##

echo "###############################################################################"
echo "model_name $model_name"

translations_sub=$translations/beam4/$model_name

mkdir -p $translations_sub

CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python -m joeynmt translate $configs/$model_name.yaml < $data/test.$src > $translations_sub/test.$model_name.$trg

# remove first line with "sentence" to make hyp and ref equal length

tail -n +2 $translations_sub/test.$model_name.$trg > $translations_sub/test.$model_name.clean.$trg
mv $translations_sub/test.$model_name.clean.$trg $translations_sub/test.$model_name.$trg

# compute case-sensitive BLEU 

cat $translations_sub/test.$model_name.$trg | sacrebleu $data/test.$trg


echo "time taken:"
echo "$SECONDS seconds"
