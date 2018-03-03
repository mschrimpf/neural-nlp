import logging

_logger = logging.getLogger(__name__)


def skip_thoughts(path_to_pretrained='ressources/skip-thoughts/'):
    import skipthoughts
    model = skipthoughts.load_model(path_to_models=path_to_pretrained, path_to_tables=path_to_pretrained)
    encoder = skipthoughts.Encoder(model)
    # vectors = encoder.encode(X)
    return encoder
