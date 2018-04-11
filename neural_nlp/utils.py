import inspect
import logging
import os
import pickle


def store(storage_directory=os.path.join(os.path.dirname(__file__), '..', 'output')):
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
