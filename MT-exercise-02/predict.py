#! /usr/bin/python3

import sys
import argparse
import logging

from train import LanguageModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--languages", type=str, nargs="+", help="ISO codes for models in --models", required=True)
    parser.add_argument("--models", type=str, nargs="+", help="Paths to models to load", required=True)

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    logging.basicConfig(level=logging.DEBUG)

    logging.debug(args)

    assert len(args.models) == len(args.languages)

    models = []
    predictions = {}

    text = sys.stdin.read().replace("\n", " ").strip()

    for lang, model_path in zip(args.languages, args.models):

        lm = LanguageModel(model_is_trained=True)
        lm.load(model_path)

        models.append(lm)

        prob = lm.predict(text)

        predictions[lang] = prob

    print(predictions)


if __name__ == '__main__':
    main()
