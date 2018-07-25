
# Setup

## Installation
```bash
conda env create -f environment.yml
conda activate neural-nlp
```

## Models and weights

### Skip-Thoughts
Setup NLTK
```bash
python
import nltk
nltk.download('punkt')
exit()
```

Download pre-trained model
```bash
# In the project root
mkdir -p ressources/models/skip-thoughts && cd "$_"

wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
wget http://www.cs.toronto.edu/~rkiros/models/utable.npy
wget http://www.cs.toronto.edu/~rkiros/models/btable.npy
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl

cd ../../../
```
See https://github.com/mschrimpf/skip-thoughts for more details.

### LM 1B
```bash
mkdir -p ressources/models/lm_1b && cd "$_"

wget http://download.tensorflow.org/models/LM_LSTM_CNN/graph-2016-09-10.pbtxt

wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-base
wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-char-embedding
wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-lstm
wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax0
wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax1
wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax2
wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax3
wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax4
wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax5
wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax6
wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax7
wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax8

wget http://download.tensorflow.org/models/LM_LSTM_CNN/vocab-2016-09-10.txt

cd ../../../
```
See https://github.com/tensorflow/models/tree/master/research/lm_1b.

### Transformer
```bash
mkdir -p ressources/models/transformer && cd "$_"
wget https://s3.amazonaws.com/opennmt-models/transformer-ende-wmt-pyOnmt.tar.gz
cd ../../../
```
See https://github.com/OpenNMT/OpenNMT-py

### Word2Vec
```bash
mkdir -p ressources/models/word2vec && cd "$_"
python ../../download_gdrive.py 0B7XkCwpI5KDYNlNUTTlSS21pQmM ./GoogleNews-vectors-negative300.bin.gz  # https://drive.google.com/uc?export=download&confirm=rLRy&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM
gzip -d GoogleNews-vectors-negative300.bin.gz
cd ../../../
```
See https://code.google.com/archive/p/word2vec.

### GloVe
```bash
mkdir -p ressources/models/glove && cd "$_"
wget http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip
unzip glove.840B.300d.zip -d . && rm glove.840B.300d.zip
cd ../../../
```
See https://github.com/stanfordnlp/GloVe.

### FastText
```bash
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip
```


## Stimuli
### Diverse sentences
Retrieve from https://evlab.mit.edu/sites/default/files/documents/index.html.

```bash
mkdir -p ressources/stimuli/diverse_sentences && cd "$_"
wget https://www.dropbox.com/s/jtqnvzg3jz6dctq/stimuli_384sentences.txt?dl=1
wget https://www.dropbox.com/s/qdft8l21e83kgz0/stimuli_243sentences.txt?dl=1
cd ../../../
```

### Naturalistic stories
From https://github.com/languageMIT/naturalstories.

```bash
mkdir -p ressources/stimuli/naturalistic_stories && cd "$_"
wget https://github.com/languageMIT/naturalstories/blob/master/naturalstories_RTS/all_stories.tok
cd ../../../
```
