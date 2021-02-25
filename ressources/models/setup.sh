#!/bin/bash

cd word2vec
chmod +x setup.sh
./setup.sh
cd ..

cd glove
chmod +x setup.sh
./setup.sh
cd ..

cd skip-thoughts
chmod +x setup.sh
./setup.sh
cd ..

cd lm_1b
chmod +x setup.sh
./setup.sh
cd ..
