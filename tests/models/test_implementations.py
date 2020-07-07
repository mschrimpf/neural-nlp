import numpy as np
import pytest
from brainio_base.assemblies import NeuroidAssembly

from neural_nlp.models.implementations import load_model, model_pool, model_layers


class TestActivations:
    models = list(model_pool.keys())
    model_num_layers = [(model, len(model_layers[model])) for model in models]

    @pytest.mark.parametrize("num_sentences", [1, 3, 10, 57])  # 57 because 57*num_words_in_sentence > 512 embedding
    @pytest.mark.parametrize("model, num_layers", model_num_layers)
    def test_sentences(self, model, num_layers, num_sentences):
        sentence = 'The quick brown fox jumps over the lazy dog'
        sentences = [sentence] * num_sentences
        model = load_model(model)
        activations = model(sentences, layers=model.default_layers)
        assert isinstance(activations, NeuroidAssembly)
        assert 2 == len(activations.shape)
        assert num_layers == len(np.unique(activations['layer']))
        assert num_sentences == len(activations['presentation'])
        layers = list(set(activations['layer'].values))
        base_shape = activations.sel(layer=layers[0]).shape
        assert all([activations.sel(layer=layer).shape == base_shape for layer in layers])
        assert len(activations.sel(layer=layers[0])['neuroid']) == model.features_size

    @pytest.mark.parametrize('num_words', [8])
    @pytest.mark.parametrize("model", models)
    def test_words(self, model, num_words):
        words = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
        assert num_words <= len(words)
        sentence = ' '.join(words[:num_words])
        model = load_model(model)
        activations = model([sentence], layers=model.default_layers, average_sentence=False)
        assert num_words == len(activations['presentation'])

    @pytest.mark.parametrize("model", models)
    def test_story(self, model):
        story = [
            'If you were to journey to the North of England you would come to a valley that is surrounded by moors as high as',
            'It is in this valley where you would find the city of Bradford where once a thousand spinning jennies that hummed and clattered spun wool into money for the long-bearded',
            'That all mill owners were generally busy as beavers and quite pleased with themselves for being so successful and well off was known to the residents of Bradford and if you were to go into the city to visit the stately City Hall you would see there the Crest of the City of Bradford which those same mill owners created to celebrate their achievements.',
            "It shows a sinister looking boar's head sitting on top of a well which seems puzzling at first but the reason for this symbol is a matter of legend.",
            'There was once legend has it a fearsome boar which lived in a wood located just outside the manor of Bradford',
            'A source of great trouble to the local folk the boar was bringing terror to the peaceful flocks and ravaging the countryside around.',
            'Even worse however the boar most liked to go to the well that was in the wood and drink its fresh water so that the people of Bradford had second thoughts about',
            "That the people of Bradford bore the brunt of the beast's ferocity was unfair in the eyes of",
            'Eventually the issue reached the ears of the kindly Lord of the Manor who the people had often',
            'The Lord saw the severity of the problem the people faced and suggested a contest',
            'He said that whoever could kill the boar and bring as proof its head to the Manor House would be rewarded with land and fame.',
            'It was the people of Bradford and the people who knew them who rejoiced at this proclamation but one question remained: who would kill the boar?',
            'By the handsome reward many felt tempted but the thought of the boar with its deadly tusks and face like thunder soon put an end to their ambitions',
            "However there was one huntsman who was still wet behind the ears who decided the prize was worth a shot in spite of the boar's reputation.",
            'The huntsman discovered the boar preferred to come out in the middle of the day.',
            'So he went to the wood by the well with his good bow to bide his time.',
            'Around noontime the boar feared by the locals came out as slow as a snail from among the trees just as the huntsman had predicted.',
            'The huntsman leaped from his hiding place and through the heart with his fine arrows',
            'Now the problem was no longer to kill the boar.', 'It was to find a way to carry the',
            'The head was too heavy for the little huntsman to carry back to the Manor House but the huntsman who was as quick-witted and cunning as ever asked himself "What else can I do to prove I killed the boar?"',
            "The next instance he opened the boar's mouth and cut out its tongue taking that",
            'He set out for the Manor House as quickly as he could and he had only been gone a few minutes when a second huntsman not so bold as the first but a little more cunning',
            'Seeing the slain carcass of the boar the huntsman rejoiced',
            "The second huntsman knew a shortcut to the Manor House was just around the nearby pond and being a bigger man than the first was able to pick up the boar's head and carry it through the wood towards the prize that awaited him.",
            'The Lord of the Manor was seated in his hall when the second huntsman burst through the door and began to spin a line.',
            "And with that at the feet of his lord he dropped the boar's", '"Then you will be rewarded as I promised."',
            '"But first let me examine the head of this monster" said the Lord of the Manor.',
            'To not examine the head in advance had not been very clever on the part of the',
            'Right away the Lord noticed the tongue of the boar was missing and exclaimed', 'he demanded.',
            'All eyes were on the huntsman.', 'The huntsman questioned by the Lord replied "I cannot say my Lord."',
            'He was suddenly realizing his tricky situation could end badly for him.',
            'Through the door at this moment the first huntsman', '"I have slain the boar the people of Bradford',
            'All eyes turned to the man who now stood in the doorway.', '"The reward is already given" said the Lord',
            '"The man here has brought to me the boar\'s head."',
            "And so saying he drew out the boar's tongue he had put in his hunting pouch and related how he had ambushed the creature in the wood and cut out its tongue as proof of his victory.",
            'Listening to the tale the Lord of the Manor tried to discern which huntsman was telling him the truth and what the two huntsmen were thinking as they waited for him to make',
            'Almost instantly the Lord saw that the second huntsman had fed him a pack of lies and it was the first huntsman who was the true savior of Bradford',
            "The Lord believed the second huntsman had tried to steal the first huntsman's prize and so proclaimed the first huntsman the true victor.",
            'The Lord wondered for a moment why he always encountered so much chicanery in his everyday',
            'The huntsman who had earned his prize fair and square received as his reward a piece of land just outside the town known',
            'His fame was indeed assured but it was not nearly as lasting as that of the fearsome Bradford Boar.',
        ]
        model = load_model(model)
        activations = model(story, model.default_layers)
        assert len(story) == len(activations['presentation'])


def test_rntn():
    _test_model('rntn', sentence='If you were to journey to the North of England '
                                 'you would come to a valley that is surrounded by moors as high as')


def test_decaNLP():
    _test_model('decaNLP')


def _test_model(model_name, sentence='The quick brown fox jumps over the lazy dog'):
    model = load_model(model_name)
    encoding = model([sentence])
    assert isinstance(encoding, np.ndarray)
    print(encoding.shape)
    assert len(encoding.shape) == 2
    assert encoding.shape[0] == 1


def test_untrained_weights_different(model_name='gpt2-xl'):
    trained = load_model(model_name)
    untrained = load_model(f"{model_name}-untrained")
    trained = trained._model.state_dict()
    untrained = untrained._model.state_dict()
    assert len(trained) == len(untrained)
    assert set(trained.keys()) == set(untrained.keys())
    weights_different = {key: trained[key] != untrained[key] for key in trained.keys()}
    assert all(weights_different)
