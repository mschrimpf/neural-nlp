
# Setup

## Models

### Skip-Thoughts
```bash
# In the project root
mkdir -p ressources/skip-thoughts && cd "$_"

wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
wget http://www.cs.toronto.edu/~rkiros/models/utable.npy
wget http://www.cs.toronto.edu/~rkiros/models/btable.npy
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl
```
See https://github.com/mschrimpf/skip-thoughts for more details.

### LM 1B
```bash
mkdir -p ressources/lm_1b && cd "$_"

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
```

### FastText
```bash
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip
```


## Data
### Diverse sentences
Retrieve from https://evlab.mit.edu/sites/default/files/documents/index.html.

```bash
mkdir -p ressources/data/diverse_sentences && cd "$_"

wget https://www.dropbox.com/s/jtqnvzg3jz6dctq/stimuli_384sentences.txt?dl=1
wget https://www.dropbox.com/s/qdft8l21e83kgz0/stimuli_243sentences.txt?dl=1
```

### Naturalistic stories
From https://github.com/languageMIT/naturalstories.

```bash
mkdir -p ressources/data/naturalistic_stories && cd "$_"

wget https://github.com/languageMIT/naturalstories/blob/master/naturalstories_RTS/all_stories.tok
```