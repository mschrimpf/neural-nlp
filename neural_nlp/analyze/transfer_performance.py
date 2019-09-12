from importlib import import_module
from torchtext.data import Field as torchtext_Field
import time

import math
import logging
import sys

import fire
import numpy as np
import torch
import torch.nn.functional as F
from io import open
from pathlib import Path
from pytorch_transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, TransfoXLCorpus
from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from pytorch_transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from pytorch_transformers import XLNetLMHeadModel, XLNetTokenizer
from torch.utils.data import DataLoader, Subset
from tqdm import trange, tqdm
from typing import Union

from result_caching import store

_logger = logging.getLogger(__name__)


@store(identifier_ignore=['device', 'batch_size'])
def language_modeling(model_identifier, data='WikiText2', device=None, batch_size=35, context_length='auto',
                      temperature=1.0, top_k=0, top_p=.9, keep_newlines=False):
    # combined from
    # https://github.com/pytorch/examples/blob/90738a76837d04e6de1403962acd21df5fbb820c/word_language_model/main.py
    # and
    # https://github.com/huggingface/pytorch-transformers/blob/df9d6effae43e92761eb92540bc45fac846789ee/examples/run_generation.py
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    _logger.debug(f"Using device {device}")

    # model
    # model = model_pool[model_identifier]
    # model.mode = BrainModel.Modes.language_modeling

    MODEL_CLASSES = {
        'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
        'gpt2-medium': (GPT2LMHeadModel, GPT2Tokenizer),
        'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
        'xlnet-large-cased': (XLNetLMHeadModel, XLNetTokenizer),
        'xlnet-base-cased': (XLNetLMHeadModel, XLNetTokenizer),
        'transfo-xl-wt103': (TransfoXLLMHeadModel, TransfoXLTokenizer),
    }
    model_class, tokenizer_class = MODEL_CLASSES[model_identifier]
    tokenizer = tokenizer_class.from_pretrained(model_identifier)
    model = model_class.from_pretrained(model_identifier)
    model = model.to(device)
    model.eval()
    if context_length == 'auto':
        context_length = model.config.max_position_embeddings
        # if context_length < 0:
        #     context_length = 1024

    def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    def sample_sequence(model, length, context_ids, labels=None,
                        temperature: Union[int, float] = 1,
                        top_k=0, top_p=0.0, is_xlnet=False, device='cpu'):
        with torch.no_grad():
            context = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
            inputs = {'input_ids': context}
            if is_xlnet:
                # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                input_ids = torch.cat((context, torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
                perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float,
                                        device=device)
                perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                target_mapping[0, 0, -1] = 1.0  # predict last token
                inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

            logits, outputs = model(**inputs, labels=labels)
            # next_token_logits = logits[0, -1, :] / temperature
            # filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            # softmax = F.softmax(filtered_logits, dim=-1)
            # softmax = softmax.unsqueeze(0)
            # predictions = softmax
            predictions = logits

        # assert predictions.shape[0] == length
        assert predictions.shape[1] == length
        return predictions

    # data
    _logger.debug(f"Loading data {data}")
    datasets = import_module('torchtext.datasets')
    dataset = getattr(datasets, data)
    test_dataset, = dataset.splits(torchtext_Field(tokenize=tokenizer.encode), train=None, validation=None,
                                   newline_eos=False)
    # from https://github.com/cybertronai/bflm/blob/b6ba6d97c9ccdf2b12e104fbdcd0bed25ada7b68/data_loader.py#L57-L60
    samples = test_dataset.examples[0].text
    chunked_dataset = Subset(np.array(samples), [
        slice(i, i + context_length)
        for i in range(0, len(samples) - (len(samples) % context_length), context_length)])
    data_loader = DataLoader(chunked_dataset, batch_size=batch_size, shuffle=True)

    # evaluate
    _logger.debug("Evaluating")
    criterion = torch.nn.CrossEntropyLoss()  # torch.nn.NLLLoss(size_average=False)
    norm_term = 0
    total_loss = 0.
    progress = tqdm(data_loader)
    for text in progress:
        out = sample_sequence(
            model=model,
            context_ids=text,
            length=len(text),
            # labels=torch.tensor(data, device=device),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
            is_xlnet=bool(model_identifier == "xlnet"),
        )
        loss = criterion(out, torch.tensor(text, device=device)).item()
        # norm_term += len(targets)
        progress.set_postfix(loss=loss, ppl=np.exp(loss))
        # total_loss += loss
        total_loss += len(text) * loss
    # test_loss = total_loss / norm_term
    test_loss = total_loss / len(samples)
    perplexity = np.exp(test_loss)
    print(f"Loss: {test_loss} | Perplexity: {perplexity}")
    return test_loss


def preserving_encode(test_data, tokenizer):
    NEWLINE_TOKEN = '....'
    test_data = test_data.replace('\n', NEWLINE_TOKEN)
    test_data = tokenizer.tokenize(test_data)
    newline_data = []
    for token in test_data:
        if NEWLINE_TOKEN in token:
            num = token.count(NEWLINE_TOKEN)
            assert num * len(NEWLINE_TOKEN) == len(token.lower().replace('ġ', '')), \
                f"inline new-lines not implemented ({token})"
            if token.lower()[0] == 'ġ':
                newline_data += token.lower()[0] + '\n'
                num -= 1
            newline_data += num * ['\n']
        else:
            newline_data.append(token)
    test_data = tokenizer.convert_tokens_to_ids(newline_data)
    return test_data


def run_transfoxl():
    # https://github.com/huggingface/pytorch-transformers/blob/df9d6effae43e92761eb92540bc45fac846789ee/examples/single_model_scripts/run_transfo_xl.py
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
    parser.add_argument('--model_name', type=str, default='transfo-xl-wt103',
                        help='pretrained model name')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size')
    parser.add_argument('--tgt_len', type=int, default=128,
                        help='number of tokens to predict')
    parser.add_argument('--ext_len', type=int, default=0,
                        help='length of the extended context')
    parser.add_argument('--mem_len', type=int, default=1600,
                        help='length of the retained previous heads')
    parser.add_argument('--clamp_len', type=int, default=1000,
                        help='max positional embedding index')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Do not use CUDA even though CUA is available')
    parser.add_argument('--no_log', action='store_true',
                        help='do not log the eval result')
    parser.add_argument('--same_length', action='store_true',
                        help='set same length attention with masking')
    args, _ = parser.parse_known_args()
    assert args.ext_len >= 0, 'extended context length must be non-negative'

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    _logger.info("device: {}".format(device))

    _logger.info("Building corpus")
    # Load a pre-processed dataset
    # You can also build the corpus yourself using TransfoXLCorpus methods
    # The pre-processing involve computing word frequencies to prepare the Adaptive input and SoftMax
    # and tokenizing the dataset
    # The pre-processed corpus is a convertion (using the conversion script )
    # corpus = TransfoXLCorpus.from_pretrained(args.model_name)
    corpus = TransfoXLCorpus()
    # corpus.vocab.add_special('<eos>')
    corpus.vocab.special.append('<eos>')
    corpus.build_corpus(Path(__file__).parent.parent.parent / 'ressources' / 'ml' / 'wikitext-103', 'wt103')

    _logger.debug("Retrieving iterator")
    te_iter = corpus.get_iterator('test', args.batch_size, args.tgt_len,
                                  device=device, ext_len=args.ext_len)

    # Load a pre-trained model
    _logger.debug("Loading model")
    model = TransfoXLLMHeadModel.from_pretrained(args.model_name)
    model = model.to(device)

    _logger.info(f'Evaluating with bsz {args.batch_size} tgt_len {args.tgt_len} ext_len {args.ext_len} '
                 f'mem_len {args.mem_len} clamp_len {args.clamp_len}')

    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    if args.clamp_len > 0:
        model.clamp_len = args.clamp_len
    if args.same_length:
        model.same_length = True

    ###############################################################################
    # Evaluation code
    ###############################################################################
    def evaluate(eval_iter):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        criterion = torch.nn.NLLLoss(size_average=False)
        total_len, total_loss = 0, 0.
        start_time = time.time()
        with torch.no_grad():
            mems = None
            for idx, (data, target, seq_len) in tqdm(enumerate(eval_iter), desc='iteration', total=eval_iter.n_batch):
                loss, _, mems = model(data, target, mems)
                # softmax_output, _ = model(data, None, mems)
                # loss = criterion(softmax_output, target)  # fails right now
                loss = loss.mean()
                total_loss += seq_len * loss.item()
                total_len += seq_len
            total_time = time.time() - start_time
        _logger.info('Time : {:.2f}s, {:.2f}ms/segment'.format(
            total_time, 1000 * total_time / (idx + 1)))
        return total_loss / total_len

    # Run on test data.
    test_loss = evaluate(te_iter)

    def format_log(loss):
        log_str = f'| loss {loss:5.2f} | ppl {math.exp(loss):9.3f} '
        return log_str

    _logger.info('=' * 100)
    _logger.info(format_log(test_loss))
    _logger.info('=' * 100)


def text_generation(prompt=None, padding_text=None, model_type='gpt2', model_name_or_path='gpt2', multinomial=False,
                    length=20, temperature=1.0, top_k=0, top_p=.9):
    """
    Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/Transformer-XL/XLNet)
    :param multinomial False to use deterministic argmax sampling of next token, True for probabilistic multinomial
    """
    # from https://github.com/huggingface/pytorch-transformers/blob/master/examples/run_generation.py

    MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

    ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in
                      (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig)), ())
    assert model_name_or_path in ALL_MODELS

    MODEL_CLASSES = {
        'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
        'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
        'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
        'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
    }
    assert model_type in MODEL_CLASSES

    # Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
    # in https://github.com/rusiaaman/XLNet-gen#methodology
    # and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
    PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
    (except for Alexei and Maria) are discovered.
    The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
    remainder of the story. 1883 Western Siberia,
    a young Grigori Rasputin is asked by his father and a group of men to perform magic.
    Rasputin has a vision and denounces one of the men as a horse thief. Although his
    father initially slaps him for making such an accusation, Rasputin watches as the
    man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
    the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
    with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

    def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    def sample_sequence(model, length, context, num_samples=1, temperature: Union[int, float] = 1,
                        top_k=0, top_p=0.0, is_xlnet=False, device='cpu'):
        context = torch.tensor(context, dtype=torch.long, device=device)
        context = context.unsqueeze(0).repeat(num_samples, 1)
        generated = context
        with torch.no_grad():
            for _ in trange(length):

                inputs = {'input_ids': generated}
                if is_xlnet:
                    # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                    # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                    input_ids = torch.cat((generated, torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
                    perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float,
                                            device=device)
                    perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                    target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                    target_mapping[0, 0, -1] = 1.0  # predict last token
                    inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

                outputs = model(
                    **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
                next_token_logits = outputs[0][0, -1, :] / temperature
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                softmax = F.softmax(filtered_logits, dim=-1)
                next_token = torch.argmax(softmax, dim=0, keepdim=True) if not multinomial \
                    else torch.multinomial(softmax, num_samples=1)
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
        return generated

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    np.random.seed(0)
    torch.manual_seed(0)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(0)

    model_class, tokenizer_class = MODEL_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    model = model_class.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()

    if length < 0 < model.config.max_position_embeddings:
        length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < length:
        length = model.config.max_position_embeddings  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop

    while True:
        raw_text = prompt if prompt else input("Model prompt >>> ")
        if model_type in ["transfo-xl", "xlnet"]:
            # Models with memory likes to have a long prompt for short inputs.
            raw_text = (padding_text if padding_text else PADDING_TEXT) + raw_text
        context_tokens = tokenizer.encode(raw_text)
        out = sample_sequence(
            model=model,
            context=context_tokens,
            length=length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
            is_xlnet=bool(model_type == "xlnet"),
        )
        out = out[0, len(context_tokens):].tolist()
        text = tokenizer.decode(out, clean_up_tokenization_spaces=True)
        print(text)
        if prompt:
            break
    return text


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    fire.Fire()
