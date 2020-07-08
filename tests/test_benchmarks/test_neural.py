import pytest
from brainio_base.assemblies import DataAssembly
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


@pytest.mark.timeout(30)  # maximum time to load -- if the assemblies are re-packaged locally, this should fail
@pytest.mark.parametrize('benchmark_identifier', [
    'Pereira2018-encoding',
    'Pereira2018-rdm',
    'Fedorenko2016v3-encoding',
    'Fedorenko2016v3nonlang-encoding',
    'Fedorenko2016v3-rdm',
    'Blank2014fROI-encoding',
    'Blank2014fROI-rdm',
])
def test_is_stored(benchmark_identifier):
    benchmark = benchmark_pool[benchmark_identifier]
    assembly = benchmark._target_assembly
    assert isinstance(assembly, DataAssembly)
    assert set(assembly.dims) == {'presentation', 'neuroid'}
    stimulus_set = assembly.stimulus_set
    assert hasattr(stimulus_set, 'name')
    ceiling = benchmark.ceiling
    assert ceiling.sel(aggregation='center') > 0
