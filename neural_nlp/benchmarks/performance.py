# coding=utf-8
import logging
import os
import pickle
import random

from brainscore.metrics import Score
from pathlib import Path

import numpy as np
import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from tqdm import tqdm, trange

from neural_nlp.models.implementations import _PytorchTransformerWrapper, BrainModel

logger = logging.getLogger(__name__)


class LMHeadModel(torch.nn.Module):
    def __init__(self, features_model, features_size, vocab_size):
        super(LMHeadModel, self).__init__()
        self.features = features_model
        self.lm_head = nn.Linear(features_size, vocab_size)

    def forward(self, input_ids, labels=None):
        with torch.no_grad():
            features = self.features(input_ids)
        lm_logits = self.lm_head(features)

        outputs = (lm_logits,) + (features,)
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, (all hidden states), (all attentions)


class TextDataset(Dataset):
    def __init__(self, model_identifier, tokenizer, file_path='train', block_size=512):
        assert os.path.isfile(file_path), f"{file_path} is not a file"
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory,
                                            f'cached_lm_{model_identifier}_{block_size}_{filename}')

        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file %s", file_path)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
            tokenized_text = np.array(tokenized_text)  # ~10 sec with numpy, ~40 hours without

            progress = tqdm(total=len(tokenized_text), desc='truncate text into blocks')
            while len(tokenized_text) >= block_size:  # Truncate in block of block_size
                block = tokenized_text[:block_size].tolist()
                self.examples.append(tokenizer.add_special_tokens_single_sentence(block))
                tokenized_text = tokenized_text[block_size:]
                progress.update(len(block))
            progress.close()
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(model, train_dataset, val_dataset, device='cuda',
          ppl_diff_threshold=1,
          train_batch_size=4, weight_decay=0.0, learning_rate=5e-5, adam_epsilon=1e-8, warmup_steps=0,
          gradient_accumulation_steps=1, num_train_epochs=30, max_grad_norm=1.0,
          logging_steps=50):
    """ Train the model """
    tb_writer = SummaryWriter()

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    # grab only head params
    lm_head_prefix = 'lm_head.'
    optim_params = [n for n, p in model.named_parameters() if n.startswith(lm_head_prefix)]
    assert optim_params, f"lm_head parameters not found in {[n for n, p in model.named_parameters()]}"
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if n in optim_params],
         'weight_decay': weight_decay},
    ]
    # features model's parameters are already disabled in LMHead.forward
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Batch size = %d", train_batch_size)
    logger.info("  Total train batch size (w. accumulation) = %d", train_batch_size * gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(num_train_epochs), desc="Epoch")
    set_seed(42)  # Added here for reproducibility (even between python 2 and 3)
    previous_val_ppl = np.inf
    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            inputs, labels = (batch, batch)
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if logging_steps > 0 and global_step % logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / logging_steps, global_step)
                    logging_loss = tr_loss

        val_ppl = evaluate(model=model, eval_dataset=val_dataset)['perplexity']
        if val_ppl < previous_val_ppl - ppl_diff_threshold:  # all good, continue
            previous_val_ppl = val_ppl
        else:  # no more improvement --> stop
            # we could load the previous checkpoint here, but won't bother since usually the loss still decreases
            break

    tb_writer.close()


def evaluate(model, eval_dataset, eval_batch_size=4, device='cuda', prefix=""):
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = batch.to(device)

        with torch.no_grad():
            outputs = model(batch, labels=batch)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "loss": eval_loss,
        "perplexity": perplexity
    }
    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    return result


class _PerformanceBenchmark:
    def __init__(self, identifier, train_data_file, val_data_file, eval_data_file):
        self.identifier = identifier
        self.train_data_file = train_data_file
        self.val_data_file = val_data_file
        self.eval_data_file = eval_data_file

    def __call__(self, model: _PytorchTransformerWrapper):
        model.mode = BrainModel.Modes.general_features
        set_seed(42)
        lm_head = LMHeadModel(model, features_size=model.features_size, vocab_size=model.vocab_size)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lm_head = lm_head.to(device)
        model._model.to(device)
        tokenizer = model.tokenizer
        block_size = tokenizer.max_len_single_sentence if hasattr(tokenizer, 'max_len_single_sentence') \
            else tokenizer.max_len
        # train
        train_dataset = TextDataset(model_identifier=model.identifier, tokenizer=tokenizer,
                                    file_path=self.train_data_file, block_size=block_size)
        val_dataset = TextDataset(model_identifier=model.identifier, tokenizer=tokenizer,
                                  file_path=self.val_data_file, block_size=block_size)
        train(model=lm_head, train_dataset=train_dataset, val_dataset=val_dataset, device=device)

        # Evaluation
        test_dataset = TextDataset(model_identifier=model.identifier, tokenizer=tokenizer,
                                   file_path=self.eval_data_file, block_size=block_size)
        test_result = evaluate(model=lm_head, eval_dataset=test_dataset, device=device)
        score = Score([test_result[key].numpy().tolist() for key in ['perplexity', 'loss']],
                      coords={'measure': ['test_perplexity', 'test_loss']}, dims=['measure'])
        score.attrs['datasets'] = {'train': self.train_data_file,
                                   'val': self.val_data_file,
                                   'test': self.eval_data_file}
        score.attrs['benchmark_identifier'] = self.identifier
        score.attrs['model_identifier'] = model.identifier
        return score


_ml_ressources_path = Path(__file__).parent.parent.parent / 'ressources' / 'ml'


def Wikitext2Benchmark():
    data_path = _ml_ressources_path / 'wikitext-2'
    return _PerformanceBenchmark(identifier='wikitext-2',
                                 train_data_file=data_path / 'train.txt',
                                 val_data_file=data_path / 'val.txt',
                                 eval_data_file=data_path / 'test.txt')


benchmark_pool = {
    'wikitext-2': Wikitext2Benchmark,
}
