#! /usr/bin/python3

# University of Zurich
# Department of Computational Linguistics
# MT Exercise 2: Data, Pre- and Postprocessing

# Authors: Zainab Aftab & Kristina Horn

import argparse
import logging
import json

import numpy as np
import math

from typing import List


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, help="Input text file", required=True)
    parser.add_argument("--model", type=str, help="Path to save model to", required=True)
    parser.add_argument("--ngram-order", type=int, help="Ngram order of language model", required=True)

    args = parser.parse_args()

    return args


class LanguageModel():

    def __init__(self, ngram_order: int = None, model_is_trained: bool = False):

        assert (ngram_order is not None) or model_is_trained

        self.ngram_order = ngram_order

        self.probs = {}

        self.model_is_trained = model_is_trained

    def train(self, text: str):
        #your code here, instead of this:
        n = self.ngram_order
        total_ngrams = 0
        
        for i in range(0, len(text), n):
            total_ngrams += 1
            if text[i:i+n] in self.probs:
                self.probs[text[i:i+n]] += 1
            else:
                self.probs[text[i:i+n]] = 1

        return self.probs

    def predict(self, text: str):
        # your code here, instead of this:
        n = self.ngram_order
        all_ngrams = []
        total_ngrams = 0
        prob = []
        
        for i in range(0, len(text), n):
            total_ngrams += 1
            all_ngrams.append(text[i:i+n])
            if text[i:i+n] in self.probs:
                # calculation
                count_of_ngram = self.probs[text[i:i+n]]
                prob.append(math.log(count_of_ngram/total_ngrams))
            else:
                prob.append(-100)

        # print(prob)
        return sum(prob)

    def load(self, file_path):
        with open(file_path, 'r') as handle:
            self.probs = json.load(handle)

        self.ngram_order = self.probs["__ORDER__"]

    def save(self, file_path):

        self.probs["__ORDER__"] = self.ngram_order

        with open(file_path, 'w') as handle:
            json.dump(self.probs, handle, ensure_ascii=False, indent=2)


def main():

    args = parse_args()

    logging.basicConfig(level=logging.DEBUG)

    logging.debug(args)

    handle = open(args.input, "r")

    text = handle.read().replace("\n", " ").strip()

    lm = LanguageModel(ngram_order=args.ngram_order, model_is_trained=False)

    lm.train(text)

    lm.save(args.model)


if __name__ == '__main__':
    main()
