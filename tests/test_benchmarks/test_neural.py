import pytest
from pytest import approx

from neural_nlp.benchmarks.neural import PereiraEncoding, PereiraRDM, \
    Fedorenko2016Encoding, Fedorenko2016AllEncoding, \
    StoriesVoxelEncoding, StoriesfROIEncoding


@pytest.mark.parametrize('benchmark_ctr, expected', [
    (PereiraEncoding, 0.292317),
    (PereiraRDM, 0.048785),
    (Fedorenko2016Encoding, .098452),
    (Fedorenko2016AllEncoding, .178473),
    (StoriesVoxelEncoding, .086608),
    (StoriesfROIEncoding, .156544),
])
def test_ceiling(benchmark_ctr, expected):
    benchmark = benchmark_ctr()
    ceiling = benchmark.ceiling
    assert ceiling.sel(aggregation='center') == approx(expected, abs=.005)
