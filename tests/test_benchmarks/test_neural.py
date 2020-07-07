import pytest
from pytest import approx

from neural_nlp.benchmarks.neural import benchmark_pool


@pytest.mark.parametrize('benchmark_identifier, expected', [
    ('Pereira2018-encoding', 0.318567),
    ('Pereira2018-rdm', 0.048785),
    ('Fedorenko2016v3-encoding', .168649),
    ('Fedorenko2016v3nonlang-encoding', .116164),
    ('Fedorenko2016v3-rdm', .11739),
    ('Blank2014fROI-encoding', .200405),
    ('Blank2014fROI-rdm', .020128),
])
def test_ceiling(benchmark_identifier, expected):
    benchmark = benchmark_pool[benchmark_identifier]
    ceiling = benchmark.ceiling
    assert ceiling.sel(aggregation='center') == approx(expected, abs=.005)
