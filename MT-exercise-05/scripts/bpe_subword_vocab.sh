#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..
merge_ops=2000
src=nl
trg=de
lang=nl-de

codes_file=${base}/data/bpe.${merge_ops}

echo "learning * joint * BPE..."
cat ${base}/data/train.${src} ${base}/data/train.$trg > ${base}/data/train.tmp
python3 -m subword_nmt.learn_bpe -s $merge_ops --total_symbols -i ${base}/data/train.tmp -o ${base}/data/${codes_file}
rm ${base}/data/train.tmp

echo "applying BPE..."
for l in ${src} ${trg}; do
    for p in train valid test; do
        python3 -m subword_nmt.apply_bpe -c ${codes_file} -i ${base}/data/${p}.${l} -o ${base}/data/${p}.bpe.${merge_ops}.${l}
    done
done

echo "DONE!!!"














# cat $base/data/train.nl $base/data/train.de | subword-nmt learn-bpe -s 2000 -t -o $base/data/codes_file
# subword-nmt apply-bpe -c $base/data/codes_file < $base/data/train.nl | subword-nmt get-vocab > $base/data/vocab_file.nl
# subword-nmt apply-bpe -c $base/data/codes_file < $base/data/train.de | subword-nmt get-vocab > $base/data/vocab_file.de

# subword-nmt apply-bpe -c $base/data/codes_file < $base/data/train.nl | subword-nmt get-vocab > $base/data/vocab_file.nl

# subword-nmt apply-bpe -c $base/data/codes_file < $base/data/train.de| subword-nmt get-vocab > $base/data/vocab_file.de

# # subword-nmt learn-joint-bpe-and-vocab --input $base/data/train.nl $base/data/train.de -s 2000 -o $base/data/codes_file --write-vocabulary $base/data/vocab_file.nl $base/data/vocab_file.de

# subword-nmt apply-bpe -c $base/data/codes_file --vocabulary $base/data/vocab_file.nl --vocabulary-threshold 50 < $base/data/train.nl > $base/data/train.BPE.nl
# subword-nmt apply-bpe -c $base/data/codes_file --vocabulary $base/data/vocab_file.de --vocabulary-threshold 50 < $base/data/train.de > $base/data/train.BPE.de

# subword-nmt apply-bpe -c $base/data/codes_file --vocabulary $base/data/vocab_file.nl --vocabulary-threshold 50 < $base/data/test.nl > $base/data/test.BPE.de

# # remove the counts from the vocab files
# cut -d ' ' -f1 $base/data/vocab_file.nl > $base/data/vocab_woc.nl
# cut -d ' ' -f1 $base/data/vocab_file.de > $base/data/vocab_woc.de







# python3 -m subword_nmt.learn_bpe -s "${merge_ops}" -i "${tmp}/train.tmp" -o "${codes_file}"