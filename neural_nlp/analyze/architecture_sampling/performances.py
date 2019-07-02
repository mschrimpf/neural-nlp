import json
import logging
import warnings

import numpy as np
from matplotlib import pyplot
from pathlib import Path
from tqdm import tqdm

_logger = logging.getLogger(__name__)


def main(data_dir):
    data_dir = Path(data_dir)
    model_dirs = list(data_dir.iterdir())
    val_losses = []
    no_finfiles = 0
    for model_dir in tqdm(model_dirs, desc='models'):
        fin_file = model_dir / 'FIN'
        if not fin_file.is_file():
            no_finfiles += 1
            continue
        text = fin_file.read_text().split('\n')
        result = json.loads(text[1])
        val_losses.append(result['val_loss'])

    if no_finfiles:
        warnings.warn(f"Found {no_finfiles} model directories without FIN file")

    val_losses = np.array(val_losses)
    val_losses = val_losses[~np.isnan(val_losses)]

    fig, axes = pyplot.subplots(nrows=2, ncols=1)
    axes[0].hist(val_losses, bins=100, log=True)

    val_losses = val_losses[val_losses < 100]
    axes[1].hist(val_losses, bins=100)

    pyplot.savefig(Path(__file__).parent / f"val_losses-{data_dir.name}.png")


if __name__ == '__main__':
    # main('/braintree/data2/active/users/msch/zoo.bck20190408-multi30k')
    main('/braintree/data2/active/users/msch/zoo.wmt17')
