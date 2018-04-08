import os
import pandas as pd

_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'ressources', 'data')


def diverse_sentences1(filepath=os.path.join(_data_dir, 'diverse_sentences', 'stimuli_384sentences.txt')):
    return _diverse_sentences(filepath)


def diverse_sentences2(filepath=os.path.join(_data_dir, 'diverse_sentences', 'stimuli_243sentences.txt')):
    return _diverse_sentences(filepath)


def _diverse_sentences(filepath):
    with open(filepath) as f:
        return f.read().splitlines()


def naturalistic_stories(filepath=os.path.join(_data_dir, 'naturalistic_stories', 'all_stories.tok'), keep_meta=False):
    data = pd.read_csv(filepath, delimiter='\t')

    def apply(words):
        sentences = []
        sentence = ''
        for word in words:
            sentence += word
            if word.endswith('.') or word.endswith(".'"):
                sentences.append(sentence)
                sentence = ''
            else:
                sentence += ' '
        return pd.DataFrame({'sentence': sentences, 'sentence_num': list(range(len(sentences)))})

    data = data.groupby('item')['word'].apply(apply).reset_index(level=0)
    data = data[['item', 'sentence_num', 'sentence']]
    if keep_meta:
        return data
    return data['sentence'].values


data_mappings = {
    'diverse1': diverse_sentences1,
    'diverse2': diverse_sentences2,
    'naturalistic_stories': naturalistic_stories
}
