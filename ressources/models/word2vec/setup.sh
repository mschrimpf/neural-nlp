#!/bin/bash

python2 ../../download_gdrive.py 0B7XkCwpI5KDYNlNUTTlSS21pQmM ./GoogleNews-vectors-negative300.bin.gz  # https://drive.google.com/uc?export=download&confirm=rLRy&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM
gzip -d GoogleNews-vectors-negative300.bin.gz

# See https://code.google.com/archive/p/word2vec.
