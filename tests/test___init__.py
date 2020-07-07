import itertools
import numpy as np
import pytest
from brainio_base.assemblies import NeuroidAssembly
from pytest import approx

from neural_nlp import score
from neural_nlp.models.wrapper.core import attach_stimulus_set_meta


class TestScore:
    models = ['word2vec', 'glove', 'skip-thoughts', 'lm_1b', 'transformer']
    stories = ['Boar', 'MatchstickSeller', 'KingOfBirds', 'Elvis', 'HighSchool']

    @pytest.mark.parametrize("model, stimulus_set", list(itertools.product(models, stories)))
    def test_story_model(self, model, stimulus_set):
        scores = score(model=model, stimulus_set='naturalistic-neural-reduced.{}'.format(stimulus_set))
        assert 'region' in scores.aggregation
        assert scores.aggregation.sel(aggregation='center').shape[0] == 44

    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_dummy_model(self, num_layers):
        model_layers = [f'dummylayer{i}' for i in range(num_layers)]

        class DummyModel:
            def __call__(self, stimuli, layers):
                assert len(layers) == 1 and layers[0] in model_layers
                assembly = NeuroidAssembly(np.ones([len(stimuli), 1]), coords={
                    'stimulus_sentence': ('presentation', stimuli['sentence']),
                    'dummy': ('presentation', ['dummy'] * len(stimuli)),
                    'neuroid_id': ('neuroid', [model_layers.index(layers[0])]),
                    'layer': ('neuroid', layers),
                }, dims=['presentation', 'neuroid'])
                assembly = attach_stimulus_set_meta(assembly, stimuli)
                return assembly

        s = score(model=f'dummy-{num_layers}layers', model_impl=DummyModel(), layers=model_layers,
                  benchmark='Pereira2018-encoding-min')
        assert s.sel(aggregation='center') == approx(0, abs=.5)

    def test_dummy_tuple_layers(self):
        model_layers = [f'dummylayer{i}' for i in range(2)]

        class DummyModel:
            def __call__(self, stimuli, layers):
                assembly = NeuroidAssembly(np.ones([len(stimuli), 1]), coords={
                    'stimulus_sentence': ('presentation', stimuli['sentence']),
                    'dummy': ('presentation', ['dummy'] * len(stimuli)),
                    'neuroid_id': ('neuroid', [model_layers.index(layers[0])]),
                    'layer': ('neuroid', layers),
                }, dims=['presentation', 'neuroid'])
                assembly = attach_stimulus_set_meta(assembly, stimuli)
                return assembly

        s = score(model=f'dummy-tuple_layers', model_impl=DummyModel(), layers=tuple(model_layers),
                  benchmark='Pereira2018-encoding-min')
        assert s.sel(aggregation='center') == approx(0, abs=.5)
