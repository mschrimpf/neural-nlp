import json
import logging
import warnings

import fire
import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot
from pathlib import Path
from tqdm import tqdm

from result_caching import store

seaborn.set()

_logger = logging.getLogger(__name__)


def plot_histograms(data_dirs=('/braintree/data2/active/users/msch/zoo.bck20190408-multi30k',
                               '/braintree/data2/active/users/msch/zoo.wmt17')):
    for data_dir in data_dirs:
        plot_histogram(data_dir)


def plot_histogram(data_dir):
    data = collect(data_dir)
    data = data[data['data'] == 'valid']
    data = data.loc[data.groupby('model_dir')['epoch'].idxmax()]
    validation_perplexities = data['ppl'].values
    validation_perplexities = validation_perplexities[~np.isnan(validation_perplexities)]

    fig, axes = pyplot.subplots(nrows=2, ncols=1)
    axes[0].hist(validation_perplexities, bins=100, log=True)
    axes[0].set_xlabel('validation perplexity')
    axes[0].set_ylabel('# architectures')

    validation_perplexities = validation_perplexities[validation_perplexities < 100]
    axes[1].hist(validation_perplexities, bins=100)
    axes[1].set_xlabel('validation perplexity')
    axes[1].set_ylabel('# architectures')

    pyplot.savefig(Path(__file__).parent / f"validation_perplexities-{Path(data_dir).name}.png")


def best_models(data_dir, top=10):
    data = collect(data_dir)
    data = data[data['data'] == 'valid']
    data = data.loc[data.groupby('model_dir')['ppl'].idxmin()]
    data = data.sort_values('ppl')
    for i, (row_index, row) in enumerate(data.iterrows()):
        if i > top:
            break
        print(f"{row['model_dir']} --> {row['ppl']}")


@store()
def collect(data_dir):
    data_dir = Path(data_dir)
    model_dirs = list(data_dir.iterdir())
    data_rows = []
    for model_dir in tqdm(model_dirs, desc='models'):
        log_file = (model_dir / 'log')
        if not log_file.exists():
            warnings.warn(f"Log file {log_file} does not exist")
            continue
        logs = log_file.read_text().split('\n')
        logs = [json.loads(log) for log in logs if log]
        for log in logs[1:]:  # ignore first log (model hyper-parameters)
            if 'epoch' not in log:
                warnings.warn(f"Ignoring log due to missing epoch value: {log}")
                continue
            if not (model_dir / f"checkpoint-epoch{log['epoch']}.pt").exists():
                warnings.warn(f"Ignoring {model_dir}/epoch {log['epoch']} because checkpoint file does not exist")
                continue
            data_rows.append({
                'model_dir': model_dir,
                'epoch': log['epoch'],
                'data': log['data'],
                'ppl': log['ppl'],
            })
    data_rows = pd.DataFrame(data_rows)
    return data_rows


if __name__ == '__main__':
    fire.Fire()
