import logging
import sys

import fire
import numpy as np
import torch
import torch.nn.functional as F
from io import open
from pathlib import Path
from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from pytorch_transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from pytorch_transformers import XLNetLMHeadModel, XLNetTokenizer
from tqdm import trange
from typing import Union

_logger = logging.getLogger(__name__)


def language_modeling(model_identifier, data='wikitext-2/test.txt', device=None, batch_size=35, max_source_len='auto',
                      temperature=1.0, top_k=0, top_p=.9, keep_newlines=False):
    # cf. https://github.com/pytorch/examples/blob/90738a76837d04e6de1403962acd21df5fbb820c/word_language_model/main.py
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    _logger.debug(f"Using device {device}")

    # model = model_pool[model_identifier]
    # model.mode = BrainModel.Modes.language_modeling

    MODEL_CLASSES = {
        'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
        'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
        'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
        'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
    }
    model_class, tokenizer_class = MODEL_CLASSES[model_identifier]
    tokenizer = tokenizer_class.from_pretrained(model_identifier)
    model = model_class.from_pretrained(model_identifier)
    model = model.to(device)
    model.eval()
    if max_source_len == 'auto':
        max_source_len = model.config.max_position_embeddings

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
                context = context_ids[len(context_ids) - step - max_source_len:len(context_ids) - step]
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
    criterion = torch.nn.NLLLoss(size_average=False)  # torch.nn.CrossEntropyLoss()

    total_loss = 0.
    for i in trange(0, len(test_data) - 1, batch_size, desc='test batches'):
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
        total_loss += len(data) * criterion(out, torch.tensor(targets, device=device)).item()
    test_loss = total_loss / (len(test_data) - 1)
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


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    fire.Fire()
