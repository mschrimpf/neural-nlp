import os

_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'ressources', 'data')


def diverse_sentences1(filepath=os.path.join(_data_dir, 'diverse_sentences', 'stimuli_384sentences.txt')):
    return _diverse_sentences(filepath)


def diverse_sentences2(filepath=os.path.join(_data_dir, 'diverse_sentences', 'stimuli_243sentences.txt')):
    return _diverse_sentences(filepath)


def _diverse_sentences(filepath):
    with open(filepath) as f:
        return f.read().splitlines()


def naturalistic_stories(filepath=os.path.join(_data_dir, 'naturalistic_stories', 'all_stories.tok')):
    with open(filepath) as f:
        next(f)  # ignore header
        words = (line.split()[0] for line in f)  # only first word
        sentences = []
        sentence = ''
        for word in words:
            sentence += word
            if word.endswith('.') or word.endswith(".'"):
                sentences.append(sentence)
                sentence = ''
            else:
                sentence += ' '
        return sentences


data_mappings = {
    'diverse1': diverse_sentences1,
    'diverse2': diverse_sentences2,
    'naturalistic_stories': naturalistic_stories
}
