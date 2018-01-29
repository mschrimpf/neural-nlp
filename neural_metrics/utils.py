import pickle


def save(image_activations, target_filepath):
    if not target_filepath.endswith('.pkl'):
        target_filepath = target_filepath + '.pkl'
    print("Saving to %s" % target_filepath)
    with open(target_filepath, 'wb') as file:
        pickle.dump(image_activations, file)
