import os
import pandas as pd

_data_dir = os.path.join(os.path.dirname(__file__), '..', 'ressources', 'stimuli')


class DiverseSentences(object):
    set_mapping = {'1': 'stimuli_384sentences.txt', '2': 'stimuli_243sentences.txt'}

    def __init__(self, stimulus_set_filename, stimuli_dir=os.path.join(_data_dir, 'diverse_sentences')):
        self._filepath = os.path.join(stimuli_dir, stimulus_set_filename)

    def __call__(self):
        with open(self._filepath) as f:
            return f.read().splitlines()


class NaturalisticStories(object):
    set_mapping = {'Boar': 1, 'Aqua': 2, 'MatchstickSeller': 3, 'KingOfBirds': 4, 'Elvis': 5,
                   'MrSticky': 6, 'HighSchool': 7, 'Roswell': 8, 'Tulips': 9, 'Tourette': 10}

    def __init__(self, stimulus_set_name,
                 stimuli_filepath=os.path.join(_data_dir, 'naturalistic_stories', 'all_stories.tok')):
        self._stimulus_set = self.set_mapping[stimulus_set_name]
        self._filepath = stimuli_filepath

    def __call__(self, keep_meta=False):
        data = pd.read_csv(self._filepath, delimiter='\t')

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
        data = data[data['item'] == self._stimulus_set]
        if keep_meta:
            return data
        return data['sentence'].values


def load_stimuli(stimulus_set_name):
    return _mappings[stimulus_set_name]()


_mappings = {
    **{'diverse{}'.format(name): DiverseSentences(filename) for name, filename in DiverseSentences.set_mapping.items()},
    **{'naturalistic{}'.format(name): NaturalisticStories(name) for name in NaturalisticStories.set_mapping}}
