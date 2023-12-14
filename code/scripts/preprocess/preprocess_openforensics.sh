#!/bin/bash

python -m datasets.OpenForensics.preprocess --dset-path $1 --mode train
python -m datasets.OpenForensics.preprocess --dset-path $1 --mode val 
python -m datasets.OpenForensics.preprocess --dset-path $1 --mode test 
