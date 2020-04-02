import itertools
import logging
import matplotlib
import seaborn
import sys

from neural_nlp.analyze import scores
from neural_nlp.analyze.data import ceiling
from neural_nlp.analyze.scores import bars, layers

_logger = logging.getLogger(__name__)


def paper_figures():
    seaborn.set(context='talk')
    seaborn.set_style("whitegrid", {'axes.grid': False})
    matplotlib.rc('axes', edgecolor='black')
    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.spines.top'] = False

    neural_prefixes = ['Pereira2018', 'Fedorenko2016', 'Blank2014', 'Futrell2018', 'overall']
    neural_benchmarks = [f'{prefix}-encoding' for prefix in neural_prefixes]
    # 1: we can predict
    _logger.info("Figures 1")
    bars.fmri_best()
    bars.stories_best()
    bars.ecog_best()
    bars.overall()
    # 2: scores are correlated
    _logger.info("Figures 2")
    for best_layer in [True, False]:
        bars.benchmark_correlations(best_layer=best_layer)  # aggregates
        scores.fmri_experiment_correlations(best_layer=best_layer)  # within Pereira
        for benchmark1, benchmark2 in itertools.combinations(neural_benchmarks, 2):
            scores.compare(benchmark1=benchmark1, benchmark2=benchmark2, best_layer=best_layer)
    # 3: LM predicts brain
    _logger.info("Figures 3")
    for neural_benchmark in neural_benchmarks:
        scores.compare(benchmark1='wikitext-2', benchmark2=neural_benchmark, best_layer=True)
    # 4: neural/LM predicts behavior
    scores.compare(benchmark1='Pereira2018-encoding', benchmark2='Futrell2018-encoding', best_layer=True)
    scores.compare(benchmark1='wikitext-2', benchmark2='Futrell2018-encoding', best_layer=True)
    # 5: untrained predicts trained
    _logger.info("Figures 4")
    for neural_benchmark in neural_benchmarks:
        scores.untrained_vs_trained(benchmark=neural_benchmark)
    # S1: cross-metrics
    _logger.info("Figures S1")
    for benchmark_prefix in neural_prefixes:
        scores.compare(benchmark1=f"{benchmark_prefix}-encoding", benchmark2=f"{benchmark_prefix}-rdm")
    # S2: non language signal
    _logger.info("Figures S2")
    scores.compare(benchmark1='Fedorenko2016v2-encoding', benchmark2='Fedorenko2016nonlangv2-encoding',
                   identity_line=True)
    scores.Pereira_language_vs_other()
    # S3: ceiling extrapolation
    _logger.info("Figures S3")
    for benchmark in neural_benchmarks:
        ceiling.plot_extrapolation_ceiling(benchmark=benchmark)
    # S4: layers
    _logger.info("Figures S4")
    for benchmark in neural_benchmarks:
        layers.layer_preference(benchmark=benchmark)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    paper_figures()
