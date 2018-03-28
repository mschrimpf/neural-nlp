def diverse_sentences1(filepath='ressources/data/stimuli_384sentences.txt'):
    return _diverse_sentences(filepath)


def diverse_sentences2(filepath='ressources/data/stimuli_243sentences.txt'):
    return _diverse_sentences(filepath)


def _diverse_sentences(filepath):
    with open(filepath) as f:
        return f.readlines()


data_mappings = {
    'diverse1': diverse_sentences1,
    'diverse2': diverse_sentences2
}
