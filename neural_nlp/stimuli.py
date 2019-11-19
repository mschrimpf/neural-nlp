import os

import pandas as pd
from pathlib import Path


class StimulusSet(pd.DataFrame):
    _metadata = ["name"]

    @property
    def _constructor(self):
        return StimulusSet


_data_dir = Path(os.path.dirname(__file__)) / '..' / 'ressources' / 'stimuli'


class DiverseSentences:
    set_mapping = {'1': 'stimuli_384sentences.txt', '2': 'stimuli_243sentences.txt'}

    def __init__(self, stimulus_set_filename, stimuli_dir=_data_dir / 'diverse_sentences'):
        self._filepath = os.path.join(stimuli_dir, stimulus_set_filename)

    def __call__(self):
        with open(self._filepath) as f:
            return f.read().splitlines()


class NaturalisticStories:
    story_item_mapping = {'Boar': 1, 'Aqua': 2, 'MatchstickSeller': 3, 'KingOfBirds': 4, 'Elvis': 5,
                          'MrSticky': 6, 'HighSchool': 7, 'Roswell': 8, 'Tulips': 9, 'Tourette': 10}
    item_story_mapping = {item: story for story, item in story_item_mapping.items()}
    sentence_end = ['.', '?', '!', ".'", "?'", "!'"]

    def __init__(self, stimuli_filepath=_data_dir / 'naturalistic_stories' / 'all_stories.tok'):
        self._filepath = stimuli_filepath

    def __call__(self, keep_meta=True):
        data = pd.read_csv(self._filepath, delimiter='\t')

        def words_to_sentences(words):
            sentences = []
            sentence = ''
            for word in words:
                sentence += word
                if any(word.endswith(sentence_end) for sentence_end in self.sentence_end):
                    sentences.append(sentence)
                    sentence = ''
                else:
                    sentence += ' '
            return pd.DataFrame({'sentence': sentences, 'sentence_num': list(range(len(sentences)))})

        data = data.groupby('item')['word'].apply(words_to_sentences).reset_index(level=0)
        data['story'] = [self.item_story_mapping[item] for item in data['item']]
        data = data[['story', 'item', 'sentence_num', 'sentence']]
        data = StimulusSet(data)
        data.name = 'naturalistic_stories'
        return data


class NaturalisticStoriesNeural:
    def __init__(self, stimulus_set_name, reduced=False):
        self._stimulus_set_name = stimulus_set_name
        self._reduced = reduced

    def __call__(self):
        from neural_nlp.neural_data.fmri import load_sentences_meta
        sentences_meta = load_sentences_meta(self._stimulus_set_name)
        return (sentences_meta['reducedSentence'] if self._reduced else sentences_meta['fullSentence']).dropna().values


class ConceptSimilarity:
    def __call__(self):
        with open(os.path.join(_data_dir, 'concept-similarity', 'words.txt')) as f:
            words = f.read().splitlines()
        return words


def load_stimuli(stimulus_set_name):
    return _mappings[stimulus_set_name]()


_mappings = {
    **{'diverse.{}'.format(name): DiverseSentences(filename)
       for name, filename in DiverseSentences.set_mapping.items()},
    **{'naturalistic.{}'.format(name): NaturalisticStories(name)
       for name in NaturalisticStories.story_item_mapping},
    **{'naturalistic-neural-full.{}'.format(name): NaturalisticStoriesNeural(name, reduced=False)
       for name in NaturalisticStories.story_item_mapping},
    **{'naturalistic-neural-reduced.{}'.format(name): NaturalisticStoriesNeural(name, reduced=True)
       for name in NaturalisticStories.story_item_mapping},
    **{'concept-similarity': ConceptSimilarity()}}
