#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..


# Set the input and output file paths
input_source_file=$base/data/train_long.nl
output_source_file=$base/data/train.nl

input_target_file=$base/data/train_long.de
output_target_file=$base/data/train.de

# Defining the number of lines to extract
num_lines=100000

# Extract number of lines from source file
head -n $num_lines $input_source_file > $output_source_file

# Extract number of lines from target file
head -n $num_lines $input_target_file > $output_target_file
