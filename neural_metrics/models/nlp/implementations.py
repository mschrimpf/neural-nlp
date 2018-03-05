import numpy as np


def skip_thoughts(path_to_pretrained='ressources/skip-thoughts/'):
    """
    http://papers.nips.cc/paper/5950-skip-thought-vectors
    """
    import skipthoughts
    model = skipthoughts.load_model(path_to_models=path_to_pretrained, path_to_tables=path_to_pretrained)
    encoder = skipthoughts.Encoder(model)
    return encoder.encode


def lm_1b():
    """
    https://arxiv.org/pdf/1602.02410.pdf
    """
    from lm_1b.lm_1b_eval import Encoder

    encoder = Encoder(vocab_file='ressources/lm_1b/vocab-2016-09-10.txt',
                      pbtxt='ressources/lm_1b/graph-2016-09-10.pbtxt',
                      ckpt='ressources/lm_1b/ckpt-*')

    def encode(sentences):
        embeddings, word_ids = encoder(sentences)
        return np.array([embedding[-1][0] for embedding in embeddings])  # only output last embedding, discard time

    return encode
