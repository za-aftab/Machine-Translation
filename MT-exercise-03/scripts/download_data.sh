#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data

mkdir -p $data

tools=$base/tools

# link default training data for easier access

mkdir -p $data/wikitext-2

for corpus in train valid test; do
    absolute_path=$(realpath $tools/pytorch-examples/word_language_model/data/wikitext-2/$corpus.txt)
    ln -snf $absolute_path $data/wikitext-2/$corpus.txt
done

# download a different interesting data set!

mkdir -p $data/decleration

mkdir -p $data/decleration/raw

wget https://www.gutenberg.org/files/1/1-0.txt
mv 1-0.txt $data/decleration/raw/decleration.txt

# preprocess slightly

cat $data/decleration/raw/decleration.txt | python $base/scripts/preprocess_raw.py > $data/decleration/raw/decleration.cleaned.txt

# tokenize, fix vocabulary upper bound

cat $data/decleration/raw/decleration.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 5000 --tokenize --lang "en" --sent-tokenize > \
    $data/decleration/raw/decleration.preprocessed.txt

# split into train, valid and test

head -n 440 $data/decleration/raw/decleration.preprocessed.txt | tail -n 400 > $data/decleration/valid.txt
head -n 840 $data/decleration/raw/decleration.preprocessed.txt | tail -n 400 > $data/decleration/test.txt
tail -n 3075 $data/decleration/raw/decleration.preprocessed.txt | head -n 2955 > $data/decleration/train.txt
