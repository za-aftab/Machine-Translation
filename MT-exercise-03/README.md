# MT Exercise 3: Pytorch RNN Language Models
# Kristina Horn & Zainab Aftab

This repo shows how to train neural language models using [Pytorch example code](https://github.com/pytorch/examples/tree/master/word_language_model). Thanks to Emma van den Bold, the original author of these scripts. 

# Changes that were made:
- The download_data.sh was adjusted so that a textfile containing the Declaration of Independence of the USA was download and preprocessed.
- train.sh was changed so that it used the previously generated dataset of the US-Declaration.
-> it also now generates multiple models with diffrent dropout values (0.0, 0.1, 0.5, 0.7, 0.9)
- we added a additional flag to "/tools/pytorch-examples/word language model/main.py" with which you input the file name, where the perplexities get saved as a log-file. The Log-file is a simple txt-file with the training-, test- & validation-perplexity values. We copied those values manually into a excel table.
- added two python scripts to generate the wanted line-plots, each for the training and the validation perplexity.
lineplots_test_ppl.py creates a line plot for the test perplexity where as
lineplots_valid_ppl.py creates a line plot for the validation perplexity.


# Steps

Create a new virtualenv that uses Python 3:

    ./scripts/make_virtualenv.sh

Download and install required software:

    ./scripts/install_packages.sh

Download and preprocess data:

    ./scripts/download_data.sh

Train a model:

    ./scripts/train.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved.


Generate (sample) some text from a trained model with:

    ./scripts/generate.sh
To generate samples with the specific dropout numbers, you can manually adjust the number in the file.
Note: To choose the dropout 0.5, you would change the name 'model' to model_05'.

To generate the line plots run the lineplot-python-scripts.
