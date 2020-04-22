import logging
import matplotlib
import seaborn
import sys

from neural_nlp.analyze import scores
from neural_nlp.analyze.data import ceiling
from neural_nlp.analyze.scores import bars, layers, story_context

_logger = logging.getLogger(__name__)


def paper_figures():
    seaborn.set(context='talk')
    seaborn.set_style("whitegrid", {'axes.grid': False})
    matplotlib.rc('axes', edgecolor='black')
    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.spines.top'] = False

    annotated_models = ['ETM', 'glove', 'skip-thoughts', 'transformer',
                        'bert-base-uncased', 'bert-large-uncased', 'roberta-large',
                        'xlm-mlm-en-2048', 'xlnet-large-cased',
                        't5-small', 'albert-xxlarge-v2',
                        'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']

    neural_data_identifiers = ['Pereira2018', 'Fedorenko2016v3', 'Blank2014fROI']
    neural_benchmarks = [f'{prefix}-encoding' for prefix in neural_data_identifiers]
    brain_data_identifiers = neural_data_identifiers + ['Futrell2018']
    brain_benchmarks = [f'{prefix}-encoding' for prefix in brain_data_identifiers]
    # 1b: overall LM-neural correlation
    scores.compare(benchmark1='wikitext-2', benchmark2='overall_neural-encoding', best_layer=True, normalize=True)
    # 2a: per benchmark scores -- we can predict
    _logger.info("Figures 1")
    for i, benchmark in enumerate(neural_benchmarks):
        bars.whole_best(benchmark=benchmark, annotate=i == 0)
    # 2b: scores are correlated
    _logger.info("Figures 2")
    scores.Pereira2018_experiment_correlations(best_layer=True, plot_correlation=False)  # within Pereira
    for comparison_benchmark in neural_benchmarks[1:]:
        scores.compare(benchmark1=neural_benchmarks[0], benchmark2=comparison_benchmark,
                       best_layer=True, plot_ceiling=False, plot_correlation=False, identity_line=True)
    # 3: LM predicts brain
    _logger.info("Figures 3")
    for neural_benchmark in neural_benchmarks:
        scores.compare(benchmark1='wikitext-2', benchmark2=neural_benchmark, best_layer=True, annotate=annotated_models,
                       plot_ceiling=False, plot_significance_stars=False)
    # 4: neural/LM predicts behavior
    scores.compare(benchmark1='overall_neural-encoding', benchmark2='Futrell2018-encoding',
                   best_layer=True, annotate=False)
    scores.compare(benchmark1='wikitext-2', benchmark2='Futrell2018-encoding', best_layer=True, annotate=False)
    # 5: untrained predicts trained
    _logger.info("Figures 4")
    for neural_benchmark in brain_benchmarks:
        scores.untrained_vs_trained(benchmark=neural_benchmark)
    # S1: cross-metrics
    _logger.info("Figures S1")
    for benchmark_prefix in brain_data_identifiers:
        scores.compare(benchmark1=f"{benchmark_prefix}-encoding", benchmark2=f"{benchmark_prefix}-rdm")
    # S2: non language signal
    _logger.info("Figures S2")
    scores.compare(benchmark1='Fedorenko2016v3-encoding', benchmark2='Fedorenko2016v3nonlang-encoding',
                   identity_line=True, plot_ceiling=False)
    scores.Pereira_language_vs_other()
    # S3: ceiling extrapolation
    _logger.info("Figures S3")
    for benchmark in brain_benchmarks:
        ceiling.plot_extrapolation_ceiling(benchmark=benchmark)
    # S4: story context
    _logger.info("Figures S4")
    story_context.plot_num_sentences(model='gpt2-xl')
    # S5: layers
    _logger.info("Figures S5")
    for benchmark in brain_benchmarks:
        layers.layer_preference(benchmark=benchmark)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    paper_figures()
