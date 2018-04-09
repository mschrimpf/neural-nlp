from neural_nlp.stimuli import mappings


def test_diverse_sentences1():
    data = mappings['diverse1']()
    assert len(data) == 384
    assert data[0] == 'An accordion is a portable musical instrument with two keyboards.'
    assert data[1] == 'A woman has different reproductive organs than a man.'


def test_diverse_sentences2():
    data = mappings['diverse2']()
    assert len(data) == 243
    assert data[0] == 'Beekeeping encourages the conservation of local habitats.'
    assert data[-1] == "This has led to more injuries, particularly to ligaments in the skier's knee."


def test_naturalistic_stories():
    data = mappings['naturalistic_stories']()
    assert len(data) == 454
    assert data[0] == 'If you were to journey to the North of England, ' \
                      'you would come to a valley that is surrounded by moors as high as mountains.'
    assert data[-1] == "The hope is that this research leads to better diagnostic tools " \
                       "and better treatments for Tourette's."
