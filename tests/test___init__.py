import numpy as np

from neural_nlp import neural_data, run, run_searchlight_search


class TestRun:
    def test_boar_skip_thoughts(self):
        region_scores = run(model='skip-thoughts', dataset_name='naturalisticBoar')
        assert 'region' in region_scores
        assert 'spotlight_start_max' in region_scores
        np.testing.assert_array_equal(region_scores.shape, [44])


class TestSpotlightSearch:
    def test_beginning(self):
        data, model_mock = self._prepare_data(0, 5)
        scores = run_searchlight_search(model_mock, data)
        scores = scores.mean(dim='region')
        assert scores.argmax(dim='searchlight_start') == 0

    def test_middle(self):
        data, model_mock = self._prepare_data(model_start=20, model_length=10)
        scores = run_searchlight_search(model_mock, data)
        scores = scores.mean(dim='region')
        assert scores.argmax(dim='searchlight_start') == 20

    def _prepare_data(self, model_start, model_length):
        data = neural_data.load_rdms('Boar')
        del data['story']
        del data['roi_low']
        del data['roi_high']
        data = data.mean(dim='subject')
        model_mock = data.sel(timepoint=list(range(model_start, model_start + model_length))).copy() \
            .rename({'timepoint': 'stimulus'})
        return data, model_mock
