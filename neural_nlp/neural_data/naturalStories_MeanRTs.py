import os
from glob import glob
from pathlib import Path

import logging
import numpy as np
import scipy.io as sio
import xarray as xr

from neural_nlp.stimuli import StimulusSet

_logger = logging.getLogger(__name__)


# @store() Disable the storing, since it is a very small file
@store()
def load_naturalStories_MeanRTs():
    # MANUAL DIR: To give access to other people running it from different directories
    ressources_dir = Path(__file__).parent.parent.parent / 'ressources'
    data_file = ressources_dir / 'neural_data' / 'naturalstories_RTS' / 'stories_data.csv'
    _logger.info(f'Data file: {data_file}')

    data = pd.read_csv(data_file)

    df_stimulus_set = data.drop('meanItemRT', 1)
    df_stimulus_set.name = 'naturalStories_MeanRTs'

    # xarray
    assembly = xr.DataArray(np.array(data['meanItemRT']),
                            dims=('presentation',),
                            coords={'word': ('presentation', np.array(data['word'])),
                                    'story_id': ('presentation', np.array(data['item'])),
                                    'word_id': ('presentation', np.array(data['zone'])),
                                    })

    # Add the stimulus_set dataframe
    assembly.attrs['stimulus_set'] = df_stimulus_set

    return assembly

if __name__ == '__main__':
    data = load_naturalStories_MeanRTs()
