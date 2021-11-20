import re

import numpy as np
import pandas as pd
import seaborn
from brainio.assemblies import walk_coords
from matplotlib import pyplot
from pathlib import Path

from brainscore.metrics import Score
from neural_nlp import benchmark_pool, model_pool, FixedLayer
from neural_nlp.analyze import savefig
from neural_nlp.models.implementations import ETM


class prefixdict(dict):
    def __init__(self, default=None, **kwargs):
        super(prefixdict, self).__init__(**kwargs)
        self._default = default

    def __getitem__(self, item):
        subitem = item
        while len(subitem) > 1:
            try:
                return super(prefixdict, self).__getitem__(subitem)
            except KeyError:
                subitem = subitem[:-1]
        return self._default


subject_columns = prefixdict(default='subject_id',
                             Fedorenko='subject_UID',
                             Pereira='subject',
                             Blank='subject_UID')


def train_test_overlap(benchmark_identifier, unique=False, do_print=False):
    benchmark = benchmark_pool[benchmark_identifier]
    stimulus_set = benchmark._target_assembly.stimulus_set

    # idea: change the metric's apply function to output train/test
    result = []

    def output_apply(*args):
        for stimulus_type, assembly in zip(["source_train", "target_train", "source_test", "target_test"], args):
            if stimulus_type.startswith("target_"):
                # ignore duplicates from "right" side of split
                continue
            presentation_values = {
                **{"type": stimulus_type},
                **{key: value for key, dims, value in walk_coords(assembly['presentation'])},
                "stimulus_lookup": [
                    stimulus_set['sentence' if hasattr(stimulus_set, 'sentence') else 'word']
                    [stimulus_set['stimulus_id'] == stimulus_id].values
                    for stimulus_id in assembly['stimulus_id'].values]}
            if benchmark_identifier.startswith('Blank2014'):
                presentation_values["stimulus_lookup"] = [line for arr in presentation_values["stimulus_lookup"]
                                                          for line in arr]
                presentation_values = {key: str(value) for key, value in presentation_values.items()}

            result.append(presentation_values)
        return Score([0],
                     coords={**{"neuroid_id": ("neuroid", [0])},
                             **{subject_column: ("neuroid", [0]) for subject_column in set(subject_columns.values())}},
                     dims=["neuroid"])

    if hasattr(benchmark, "_single_metric"):
        benchmark._single_metric.apply = output_apply
    else:
        benchmark._metric.apply = output_apply

    # run with dummy model to invoke metric
    dummy_model = model_pool['ETM']
    candidate = FixedLayer(dummy_model, ETM.available_layers[0])
    benchmark(candidate)
    result = pd.DataFrame(result)

    # plot -- we're relying on the implicit ordering of train followed by test
    stimuli_key = 'sentence' if not any(benchmark_identifier.startswith(word_benchmark)
                                        for word_benchmark in ['Fedorenko2016', 'Futrell2018']) else 'word'
    train_stimuli = result[stimuli_key][result['type'] == 'source_train'].values
    test_stimuli = result[stimuli_key][result['type'] == 'source_test'].values
    assert len(train_stimuli) == len(test_stimuli)
    overlaps = []
    for split_index in range(len(train_stimuli)):
        split_train = train_stimuli[split_index]
        split_test = test_stimuli[split_index]
        split_train = [re.sub(r'[^\w\s]', '', sentence) for sentence in split_train]
        split_test = [re.sub(r'[^\w\s]', '', sentence) for sentence in split_test]
        train_words = [word for sentence in split_train for word in sentence.split(' ') if word]
        test_words = [word for sentence in split_test for word in sentence.split(' ') if word]
        for ngram in [1, 2, 3]:
            train_ngrams = list(zip(*[train_words[i:] for i in range(ngram)]))
            test_ngrams = list(zip(*[test_words[i:] for i in range(ngram)]))
            if unique:
                train_ngrams = np.unique(train_ngrams)
                test_ngrams = np.unique(test_ngrams)
            overlap = [test_ngram for test_ngram in test_ngrams if test_ngram in train_ngrams]
            overlaps.append({'split': split_index, 'ngram': ngram,
                             'overlap': 100 * len(overlap) / len(test_words)})
    overlaps = pd.DataFrame(overlaps)
    # plot
    fig, ax = pyplot.subplots(figsize=[4, 4.8])
    seaborn.barplot(data=overlaps, x='ngram', y='overlap', facecolor='gray', ax=ax)
    ax.set_title(benchmark_identifier)
    ax.set_ylabel('train/test overlap [%]')
    savefig(fig, f'train_test_overlap-{benchmark_identifier}' + ('-unique' if unique else ''))

    # print
    if do_print:
        # excel cannot open csvs with cells >32k
        result = result.apply(lambda x: x.str.slice(0, 32_000).replace('\n', ''))
        print(result)
        result.to_csv(Path(__file__).parent / f'splits-{benchmark_identifier}.csv')
