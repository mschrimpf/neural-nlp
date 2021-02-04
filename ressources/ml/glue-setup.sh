#!/bin/bash

# Download resource download file
wget https://raw.githubusercontent.com/nyu-mll/GLUE-baselines/master/download_glue_data.py

# Get dev_ids.tsv file (used for splitting MRPC data) from external
# (The download_glue_data.py script requires file dev_ids.tsv, but it's no longer downloaded with the script.)
mkdir glue
cd glue
mkdir MRPC
cd MRPC
wget https://dl.fbaipublicfiles.com/glue/data/mrpc_dev_ids.tsv
mv mrpc_dev_ids.tsv dev_ids.tsv
cd ../..

# Delete unwanted lines from download_glue_data.py, which try to access the dev_ids.tsv file remotely
sed -i.bak -e '85,89d' download_glue_data.py

# Download GLUE benchmark resources
python download_glue_data.py --data_dir glue --tasks all

# Remove downloading scripts
rm download_glue_data.py*

# Source: https://github.com/nyu-mll/GLUE-baselines
