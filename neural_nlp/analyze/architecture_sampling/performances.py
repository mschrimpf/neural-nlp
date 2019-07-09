import json
import logging
import warnings

import numpy as np
from matplotlib import pyplot
from pathlib import Path
from tqdm import tqdm

import seaborn
seaborn.set()

_logger = logging.getLogger(__name__)


def main(data_dir):
    data_dir = Path(data_dir)
    model_dirs = list(data_dir.iterdir())
    validation_perplexities = []
    no_result = 0
    for model_dir in tqdm(model_dirs, desc='models'):
        log_file = (model_dir / 'log')
        if not log_file.exists():
            warnings.warn(f"Log file {log_file} does not exist")
            continue
        logs = log_file.read_text().split('\n')
        logs = [json.loads(log) for log in logs if log]
        val_ppls = [log for log in logs if log['data'] == 'valid']
        if not val_ppls:
            no_result += 1
            continue
        for val_ppl in val_ppls:
            epoch = val_ppl['epoch']
            if not (model_dir / f"checkpoint-epoch{epoch}.pt").exists():
                continue
            validation_perplexities.append(val_ppl['ppl'])

    if no_result:
        warnings.warn(f"Found {no_result} logs without validation perplexity")

    validation_perplexities = np.array(validation_perplexities)
    validation_perplexities = validation_perplexities[~np.isnan(validation_perplexities)]

    fig, axes = pyplot.subplots(nrows=2, ncols=1)
    axes[0].hist(validation_perplexities, bins=100, log=True)
    axes[0].set_xlabel('validation perplexity')
    axes[0].set_ylabel('# architectures')

    validation_perplexities = validation_perplexities[validation_perplexities < 100]
    axes[1].hist(validation_perplexities, bins=100)
    axes[1].set_xlabel('validation perplexity')
    axes[1].set_ylabel('# architectures')

    pyplot.savefig(Path(__file__).parent / f"validation_perplexities-{data_dir.name}.png")


if __name__ == '__main__':
    main('/braintree/data2/active/users/msch/zoo.bck20190408-multi30k')
    main('/braintree/data2/active/users/msch/zoo.wmt17')
