import pytest
from pytest import approx

from neural_nlp.benchmarks.neural import Fedorenko2016Encoding


@pytest.mark.parametrize('benchmark_ctr, expected', [
    (Fedorenko2016Encoding, .125773),
])
def test_ceiling(benchmark_ctr, expected):
    benchmark = benchmark_ctr()
    ceiling = benchmark.ceiling
    assert ceiling.sel(aggregation='center') == approx(expected, abs=.005)
