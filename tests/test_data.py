from neural_nlp.data import data_mappings


def test_diverse_sentences1():
    data = data_mappings['diverse1']()
    assert 384 == len(data)
    assert 'An accordion is a portable musical instrument with two keyboards.' == data[0]
    assert 'A woman has different reproductive organs than a man.' == data[-1]


def test_diverse_sentences2():
    data = data_mappings['diverse2']()
    assert 243 == len(data)
    assert 'Beekeeping encourages the conservation of local habitats.' == data[0]
    assert "This has led to more injuries, particularly to ligaments in the skier's knee." == data[-1]


def test_naturalistic_stories():
    data = data_mappings['naturalistic_stories']()
    assert 454 == len(data)
    assert 'If you were to journey to the North of England, ' \
           'you would come to a valley that is surrounded by moors as high as mountains.' == data[0]
    assert "The hope is that this research leads to better diagnostic tools " \
           "and better treatments for Tourette's." == data[-1]
