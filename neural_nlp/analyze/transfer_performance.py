import logging
import sys

import fire
import numpy as np
import torch
import torch.nn.functional as F
from io import open
from pathlib import Path
from pytorch_transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig
from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from pytorch_transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from pytorch_transformers import XLNetLMHeadModel, XLNetTokenizer
from tqdm import trange
from typing import Union

from result_caching import store

_logger = logging.getLogger(__name__)


@store(identifier_ignore=['device', 'batch_size'])
def language_modeling(model_identifier, data='wikitext-2/test.txt', device=None, batch_size=35, max_source_len='auto',
                      temperature=1.0, top_k=0, top_p=.9, keep_newlines=False):
    # combined from
    # https://github.com/pytorch/examples/blob/90738a76837d04e6de1403962acd21df5fbb820c/word_language_model/main.py
    # and
    # https://github.com/huggingface/pytorch-transformers/blob/df9d6effae43e92761eb92540bc45fac846789ee/examples/run_generation.py
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    _logger.debug(f"Using device {device}")

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
    if max_source_len == 'auto':
        max_source_len = model.config.max_position_embeddings
        if max_source_len < 0:
            max_source_len = 1024

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

    def sample_sequence(model, length, context_ids, temperature: Union[int, float] = 1,
                        top_k=0, top_p=0.0, is_xlnet=False, device='cpu'):
        predictions = None
        with torch.no_grad():
            # At each step, feed in all the context up to and excluding that step, and keep track of predictions
            for step in reversed(range(length)):
                context = context_ids[max(0, len(context_ids) - step - max_source_len):len(context_ids) - step]
                context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
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

                outputs = model(**inputs)
                next_token_logits = outputs[0][0, -1, :] / temperature
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                softmax = F.softmax(filtered_logits, dim=-1)
                softmax = softmax.unsqueeze(0)
                predictions = softmax if predictions is None else torch.cat((predictions, softmax), dim=0)

        assert predictions.shape[0] == length
        return predictions

    # data
    data_path = Path(__file__).parent.parent.parent / 'ressources' / 'ml' / data
    _logger.debug(f"Loading data from {data_path}")
    assert data_path.exists(), f"{data_path} does not exist"
    with open(data_path, 'r', encoding="utf8") as f:
        test_data = '\n'.join(f)

    def get_batch(source, i):
        seq_len = min(batch_size, len(source) - 1 - i)
        # we do not have to strictly limit the data to `max_source_len` here,
        # but it will avoid unnecessarily moving data to GPU
        data_start = i if max_source_len is None else max(0, i - max_source_len)
        data = source[data_start:i + seq_len]
        target = source[i + 1:i + 1 + seq_len]
        return data, target

    # We need to tokenize outside the model so that we can operate on the tokens directly for prediction.
    # Otherwise, the model will output tokens that we would have to compare against target *words*.
    if keep_newlines:
        test_data = preserving_encode(test_data, tokenizer)
    else:
        test_data = tokenizer.encode(test_data)

    criterion = torch.nn.NLLLoss(size_average=False)
    norm_term = 0
    total_loss = 0.
    progress = trange(0, len(test_data) - 1, batch_size, desc='test batches')
    for i in progress:
        data, targets = get_batch(test_data, i)
        out = sample_sequence(
            model=model,
            context_ids=data,
            length=len(targets),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
            is_xlnet=bool(model_identifier == "xlnet"),
        )
        loss = criterion(out, torch.tensor(targets, device=device)).item()
        norm_term += len(targets)
        progress.set_postfix(loss=loss, ppl=np.exp(loss / norm_term))
        total_loss += loss
    test_loss = total_loss / norm_term
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
