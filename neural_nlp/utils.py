import logging
import pickle

import os
from abc import ABCMeta
import inspect


def store(storage_directory):
    _logger = logging.getLogger(__name__ + '.' + store.__name__)

    def save_storage(obj, path):
        _logger.debug("Saving to storage: {}".format(path))
        path_dir = os.path.dirname(path)
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir, exist_ok=True)
        savepath_part = path + '.filepart'
        with open(savepath_part, 'wb') as f:
            pickle.dump({'data': obj}, f)
        os.rename(savepath_part, path)

    def load_storage(path):
        assert os.path.isfile(path)
        _logger.debug("Loading from storage: {}".format(path))
        with open(path, 'rb') as f:
            return pickle.load(f)['data']

    def decorator(function):
        def wrapper(*args, **kwargs):
            call_args = inspect.getcallargs(function, *args, **kwargs)
            storage_path = os.path.join(storage_directory, function.__module__ + '.' + function.__name__,
                                        '_'.join('{}={}'.format(key, value) for key, value in call_args.items())
                                        + '.pkl')
            if os.path.isfile(storage_path):
                return load_storage(storage_path)
            result = function(*args, **kwargs)
            save_storage(result, storage_path)
            return result

        return wrapper

    return decorator


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
