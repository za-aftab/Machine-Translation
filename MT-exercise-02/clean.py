# University of Zurich
# Department of Computational Linguistics
# MT Exercise 2: Data, Pre- and Postprocessing

# Authors: Zainab Aftab & Kristina Horn

import argparse
import logging
from train import LanguageModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src-lang", type=str, help="source language de, en, nl, or it", required=True)
    parser.add_argument("--trg-lang", type=str, help="target language de, en, nl, or it", required=True)
    parser.add_argument("--src-input", type=str, help="source input file", required=True)
    parser.add_argument("--trg-input", type=str, help="target input file", required=True)
    parser.add_argument("--src-output", type=str, help="source output file", required=True)
    parser.add_argument("--trg-output", type=str, help="target output file", required=True)

    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)    

    src_output = []
    trg_output = []
    langs = ["de", "en", "it", "nl"]

    models = []
    for i in range(len(langs)):
        models.append(LanguageModel(model_is_trained=True))

    for i in range(len(models)):
        models[i].load(f"models/model.{langs[i]}")

    src_lang_i = langs.index(args.src_lang)
    trg_lang_i = langs.index(args.trg_lang)   
    
    src_input = []
    trg_input = []
    for line in open(f"noisy_data/noisy.{args.src_lang}", "r"):
        src_input.append(line.strip())
    for line in open(f"noisy_data/noisy.{args.trg_lang}", "r"):
        trg_input.append(line.strip())

    for i in range(min(len(src_input), len(trg_input))):
        pred_src = []
        pred_trg = []
        for m in models:
            pred_src.append(m.predict(src_input[i]))
            pred_trg.append(m.predict(trg_input[i]))
        
        if pred_src.index(max(pred_src)) == src_lang_i and pred_trg.index(max(pred_trg)) == trg_lang_i:
            src_output.append(src_input[i])
            trg_output.append(trg_input[i])

    with open(args.src_output, "w") as file:
        file.write(("\n").join(src_output))
    with open(args.trg_output, "w") as file:
        file.write(("\n").join(trg_output))

if __name__ == '__main__':
    main()