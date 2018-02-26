import logging
import pickle

import os
from abc import ABCMeta


def save(image_activations, target_filepath):
    if not target_filepath.endswith('.pkl'):
        target_filepath = target_filepath + '.pkl'
    print("Saving to %s" % target_filepath)
    with open(target_filepath, 'wb') as file:
        pickle.dump(image_activations, file)


class StorageCache(dict, metaclass=ABCMeta):
    """
    Keeps computed values in memory (the cache) and writes them to a file (the storage).
    """

    def __init__(self, savepath, use_cached=True):
        super(StorageCache, self).__init__()
        self._savepath = savepath
        self._logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self._setitem_from_storage = False  # marks when items are set from loading, i.e. we do not have to save
        if use_cached:
            self._load_storage()

    def __setitem__(self, key, value):
        super(StorageCache, self).__setitem__(key, value)
        if self._setitem_from_storage:
            return
        self._save_storage()

    def _save_storage(self):
        self._logger.debug("Saving to storage: {}".format(self._savepath))
        savepath_part = self._savepath + '.filepart'
        with open(savepath_part, 'wb') as f:
            pickle.dump({'data': dict(self.items())}, f)
        os.rename(savepath_part, self._savepath)

    def _load_storage(self):
        if not os.path.isfile(self._savepath):
            self._logger.debug("No storage saved yet: {}".format(self._savepath))
            return
        self._logger.debug("Loading from storage: {}".format(self._savepath))
        with open(self._savepath, 'rb') as f:
            storage = pickle.load(f)
        self._setitem_from_storage = True
        for key, value in storage['data'].items():
            self[key] = value
        self._setitem_from_storage = False
