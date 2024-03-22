# University of Zurich
# Department of Computational Linguistics
# MT Exercise 2: Data, Pre- and Postprocessing

# Authors: Zainab Aftab & Kristina Horn

import argparse
import logging
import json
from train import LanguageModel

import numpy as np

from typing import List


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lid-languages", type=str, nargs="+", help="lid language abbrevation", required=True)
    parser.add_argument("--lid-models", type=str, nargs="+", help="multiple models", required=True)
    parser.add_argument("--dev-sets", type=str, nargs="+", help="multiple language files", required=True)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)

    count_dict = {}
    accuracy_dict = {}    

    models = []
    for i in range(len(args.lid_models)):
        models.append(LanguageModel(model_is_trained=True))

    for i in range(len(models)):
        models[i].load(args.lid_models[i])

    total_acc_sum = 0
    for dev_set in args.dev_sets:
        total_sent = 0
        correct_pred = 0
        with open(dev_set, "r") as file:
            for line in file:
                sentence = line.strip()
                total_sent += 1
                pred = []
                for m in models:
                    pred.append(m.predict(sentence))

                if pred.index(max(pred)) == args.dev_sets.index(dev_set):
                    correct_pred += 1
        accuracy = correct_pred/total_sent
        total_acc_sum += accuracy


        print(f"{args.lid_languages[args.dev_sets.index(dev_set)]}: {accuracy}")

    total_accuracy = total_acc_sum/len(args.dev_sets)
    print("Total Accuracy: ", total_accuracy)


if __name__ == '__main__':
    main()
