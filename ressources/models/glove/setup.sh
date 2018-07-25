#!/bin/bash

wget http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip
unzip glove.840B.300d.zip -d . && rm glove.840B.300d.zip
# See https://github.com/stanfordnlp/GloVe.
