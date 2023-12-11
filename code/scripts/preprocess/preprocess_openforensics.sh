#!/bin/bash

python -m datasets.OpenForensics.preprocess --dset-path $1 --mode train --margin 1.3
python -m datasets.OpenForensics.preprocess --dset-path $1 --mode val --margin 1.3
python -m datasets.OpenForensics.preprocess --dset-path $1 --mode test --margin 1.3
