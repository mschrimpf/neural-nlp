from neural_nlp import neural_data, run_spotlight_search


class TestSpotlightSearch:
    def test_beginning(self):
        data, model_mock = self._prepare_data(0, 5)
        scores = run_spotlight_search(model_mock, data)
        scores = scores.mean(dim='region')
        assert scores.argmax(dim='spotlight_start') == 0

    def test_middle(self):
        data, model_mock = self._prepare_data(model_start=20, model_length=10)
        scores = run_spotlight_search(model_mock, data)
        scores = scores.mean(dim='region')
        assert scores.argmax(dim='spotlight_start') == 20

    def _prepare_data(self, model_start, model_length):
        data = neural_data.load_rdms('Boar')
        del data['story']
        del data['roi_low']
        del data['roi_high']
        data = data.mean(dim='subject')
        model_mock = data.sel(timepoint=list(range(model_start, model_start + model_length))).copy() \
            .rename({'timepoint': 'stimulus'})
        return data, model_mock
