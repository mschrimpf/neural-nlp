
# Setup

## Installation
```bash
git clone https://github.com/mschrimpf/neural-nlp.git
cd neural-nlp
pip install -e .
```
You might have to install nltk by hand / with conda.

## Retrieve models and stimuli
```bash
cd ressources
./setup.sh
cd ..
```
This script will run all `setup.sh` files for all models and all stimuli.

You can also run only a part of the setup files if you don't need all the ressources.

## Run
To score gpt2-xl on the Pereira2018-encoding benchmark:

```bash
python neural_nlp run --model gpt2-xl --benchmark Pereira2018-encoding --log_level DEBUG
```

Specify different models or benchmark by modifying this command-line call.
