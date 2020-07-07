# coding=utf-8
import random

import logging
import numpy as np
import os
import pickle
import torch
from numpy.random.mtrand import RandomState
from pathlib import Path
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from tqdm import tqdm, trange

from brainscore.metrics import Score
from brainscore.utils import LazyLoad
from neural_nlp.models.implementations import TaskModel

logger = logging.getLogger(__name__)


class LMHeadModel(torch.nn.Module):
    def __init__(self, features_size, vocab_size, embedding_weights=None):
        """
        :param embedding_weights: set to tie head weights to embedding
        """
        super(LMHeadModel, self).__init__()
        self.linear = nn.Linear(features_size, vocab_size)
        if embedding_weights is not None:
            np.testing.assert_array_equal(self.linear.weight.shape, embedding_weights.shape)
            self.linear.weight = embedding_weights

    def forward(self, features, labels=None):
        lm_logits = self.linear(features)

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
    def __init__(self, model_identifier, model, file_path, vocab_size=None, block_size=512, max_features=4000):
        assert os.path.isfile(file_path), f"{file_path} is not a file"
        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        # Tokens
        directory, filename = os.path.split(file_path)
        cached_tokens_file = os.path.join(directory, f'cached_lm_{model_identifier}_{block_size}_{filename}')
        if os.path.exists(cached_tokens_file) and os.getenv('NOSAVE', '0') != '1':
            logger.info("Loading tokens from cached file %s", cached_tokens_file)
            with open(cached_tokens_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating tokens from dataset file %s", file_path)
            self.examples = []
            tokenized_text = model.tokenize(text, vocab_size=vocab_size)
            assert tokenized_text.max() < vocab_size
            # Truncate in block of block_size
            # Especially with the small block sizes we end up using together with the
            # "feeding in context one word increments at a time", this is not ideal because the model doesn't see a lot
            # of context. But it's going to be even more compute if we maximize the context per block.
            for i in tqdm(range(0, len(tokenized_text) - block_size + 1, block_size), desc='truncate text into blocks'):
                self.examples.append(model.tokens_to_inputs(tokenized_text[i: i + block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.
            if os.getenv('NOSAVE', '0') != '1':
                logger.info("Saving tokens into cached file %s", cached_tokens_file)
                with open(cached_tokens_file, 'wb') as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Features
        cached_features_file = os.path.join(directory, f'cached_lm_features_{model_identifier}_{block_size}_{filename}')
        if os.path.exists(cached_features_file) and os.getenv('NOSAVE', '0') != '1':
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.features = pickle.load(handle)
        else:
            self.features = []
            for block in tqdm(self.examples, desc="token blocks to features"):  # pass tokens to model
                block_features = model(block)
                self.features.append(block_features)
            self.features = np.array(self.features)
            if os.getenv('NOSAVE', '0') != '1':
                logger.info("Saving features into cached file %s", cached_features_file)
                with open(cached_features_file, 'wb') as handle:
                    pickle.dump(self.features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        assert len(self.examples) == len(self.features)

        # optional subsampling
        if self.features[0].shape[-1] > max_features:
            indices = np.arange(self.features[0].shape[-1])
            rnd = RandomState(0)
            indices = rnd.choice(indices, size=max_features, replace=False)
            self.subsample = lambda features: features[:, :, indices]
        else:
            self.subsample = lambda features: features

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        features, labels = self.features[item], self.examples[item]
        features = self.subsample(features)
        return torch.tensor(features), torch.tensor(labels)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(model, train_dataset, val_dataset, device='cuda',
          ppl_diff_threshold=1,
          train_batch_size=4, weight_decay=0.0, learning_rate=5e-5, adam_epsilon=1e-8, warmup_steps=0,
          gradient_accumulation_steps=1, num_train_epochs=50, max_grad_norm=1.0,
          seed=42, logging_steps=50):
    """ Train the model """
    from transformers import AdamW, get_linear_schedule_with_warmup
    from tensorboardX import SummaryWriter
    tb_writer = SummaryWriter()

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer_grouped_parameters = [
        {'params': list(model.parameters()),
         'weight_decay': weight_decay}]
    assert len(optimizer_grouped_parameters[0]['params']) == 2, "expected only 2 paramaters for decoder (weight+bias)"
    # features model's parameters are already disabled in LMHead.forward
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

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
    set_seed(seed)  # Added here for reproducibility (even between python 2 and 3)
    previous_val_ppl = np.inf
    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        epoch_loss = 0
        for step, (batch_features, batch_tokens) in enumerate(epoch_iterator):
            batch_features = batch_features.to(device)
            batch_tokens = batch_tokens.to(device)
            model.train()
            outputs = model(batch_features, labels=batch_tokens)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            epoch_loss += loss.mean().item()
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

        epoch_loss = epoch_loss / step
        logger.debug(f"Training epoch {epoch}: loss = {epoch_loss}, perplexity = {np.exp(epoch_loss)}")
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

    for batch_features, batch_tokens in tqdm(eval_dataloader, desc="Evaluating"):
        batch_features = batch_features.to(device)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            outputs = model(batch_features, labels=batch_tokens)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss)).numpy().tolist()

    result = {
        "loss": eval_loss,
        "perplexity": perplexity
    }
    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    return result


class _PerformanceBenchmark:
    def __init__(self, identifier, train_data_file, val_data_file, eval_data_file, tied=False, block_size=64,
                 seed=42, **kwargs):
        self.identifier = identifier
        self.train_data_file = train_data_file
        self.val_data_file = val_data_file
        self.eval_data_file = eval_data_file
        self.tied = tied
        self.block_size = block_size
        self.seed = seed
        self.kwargs = kwargs

    def __call__(self, model: TaskModel):
        model.mode = TaskModel.Modes.tokens_to_features
        set_seed(self.seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.debug(f"Using block size {self.block_size} for {model.identifier}")

        # Data
        vocab_size = min(model.vocab_size, 250000)
        train_tokens = TextDataset(model_identifier=model.identifier, model=model, block_size=self.block_size,
                                   vocab_size=vocab_size, file_path=self.train_data_file)
        val_tokens = TextDataset(model_identifier=model.identifier, model=model, block_size=self.block_size,
                                 vocab_size=vocab_size, file_path=self.val_data_file)
        test_tokens = TextDataset(model_identifier=model.identifier, model=model, block_size=self.block_size,
                                  vocab_size=vocab_size, file_path=self.eval_data_file)

        # Decoder
        logger.info(f"Vocab size: {vocab_size}")
        features_sample, _ = train_tokens[0]
        lm_head = LMHeadModel(features_size=features_sample.shape[-1], vocab_size=vocab_size,
                              embedding_weights=model.get_embedding_weights() if self.tied else None)
        lm_head = lm_head.to(device)

        # Train
        train(model=lm_head, train_dataset=train_tokens, val_dataset=val_tokens, device=device,
              seed=self.seed, **self.kwargs)

        # Evaluation
        test_result = evaluate(model=lm_head, eval_dataset=test_tokens, device=device)
        score = Score([test_result[key] for key in ['perplexity', 'loss']],
                      coords={'measure': ['test_perplexity', 'test_loss']}, dims=['measure'])
        score.attrs['datasets'] = {'train': self.train_data_file,
                                   'val': self.val_data_file,
                                   'test': self.eval_data_file}
        score.attrs['benchmark_identifier'] = self.identifier
        score.attrs['model_identifier'] = model.identifier
        return score


_ml_ressources_path = Path(__file__).parent.parent.parent / 'ressources' / 'ml'


def Wikitext2Benchmark(identifier='wikitext-2', **kwargs):
    data_path = _ml_ressources_path / 'wikitext-2'
    return _PerformanceBenchmark(identifier=identifier,
                                 train_data_file=data_path / 'train.txt',
                                 val_data_file=data_path / 'val.txt',
                                 eval_data_file=data_path / 'test.txt', **kwargs)


benchmark_pool = {
    'wikitext-2': LazyLoad(lambda: Wikitext2Benchmark(identifier='wikitext-2', tied=False, block_size=32)),
}
