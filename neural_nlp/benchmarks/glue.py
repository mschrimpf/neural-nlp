# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

import random

import json
import logging
import numpy as np
import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

from brainscore.metrics import Score
from brainscore.utils import LazyLoad
from neural_nlp.models.implementations import BrainModel

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


class DecoderHead(torch.nn.Module):
    def __init__(self, features_size, num_labels):
        super(DecoderHead, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(features_size, num_labels)

    def forward(self, features, labels=None):
        features = features.view(-1, np.prod(features.shape[1:]))
        logits = self.linear(features)

        outputs = (logits,) + (features,)
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, features


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(train_dataset, features_model, decoder_head, run_evaluation,
          train_batch_size=8, gradient_accumulation_steps=1, num_train_epochs=50, weight_decay=0,
          learning_rate=5e-5, adam_epsilon=1e-8, warmup_steps=0, max_grad_norm=1.0,
          val_diff_threshold=.01,
          seed=42, device='cuda'):
    """ Train the model """
    tb_writer = SummaryWriter()
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in decoder_head.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in decoder_head.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Batch size = %d", train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                train_batch_size * gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    previous_val_score = -np.inf
    decoder_head.zero_grad()
    train_iterator = trange(epochs_trained, int(num_train_epochs), desc="Epoch")
    set_seed(seed)  # Added here for reproductibility
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        epoch_train_loss = 0
        for step, batch in enumerate(epoch_iterator):
            decoder_head.train()
            batch = tuple(t.to(device) for t in batch)
            features = features_model(batch=batch)
            outputs = decoder_head(features, labels=batch[-1])
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()

            epoch_train_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(decoder_head.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                decoder_head.zero_grad()
                global_step += 1

        # see if we can stop early
        logs = {}
        results = run_evaluation()
        val_score = _get_val_stop_score(results)
        if val_score > previous_val_score + val_diff_threshold:  # all good, continue
            logger.info(f"validation score {val_score} > previous {previous_val_score} + {val_diff_threshold}")
            previous_val_score = val_score
        else:  # no more improvement --> stop
            logger.info(f"Early stopping in epoch {epoch}: {results}, previously {previous_val_score}")
            # we could load the previous checkpoint here, but won't bother since accuracy usually still increases
            break

        # log
        for key, value in results.items():
            eval_key = "eval_{}".format(key)
            logs[eval_key] = value

        loss_scalar = epoch_train_loss / step
        learning_rate_scalar = scheduler.get_lr()[0]
        logs["learning_rate"] = learning_rate_scalar
        logs["loss"] = loss_scalar

        for key, value in logs.items():
            tb_writer.add_scalar(key, value, global_step)
        print(json.dumps({**logs, **{"step": global_step}}))

    tb_writer.close()


def _get_val_stop_score(results):
    if 'acc' in results:  # acc,[f1,acc_and_f1]
        return results['acc']
    elif 'pearson' in results:  # pearson,spearman,corr
        return results['pearson']
    else:
        raise ValueError(f"Unknown results {results}")


def evaluate(features_model, decoder_head, task_name, eval_dataset, output_mode,
             eval_batch_size=8, device='cuda'):
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        decoder_head.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            features = features_model(batch=batch)
            labels = batch[-1]
            outputs = decoder_head(features, labels=labels)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(task_name, preds, out_label_ids)

    logger.info("***** Eval results *****")
    logger.info("  loss = %s", str(eval_loss))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    return result


def get_examples(data_dir, task, evaluate=False):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from dataset file
    logger.info("Creating features from dataset file at %s", data_dir)
    label_list = processor.get_labels()
    examples = processor.get_dev_examples(data_dir) if evaluate else processor.get_train_examples(data_dir)
    return examples, label_list, output_mode


class GLUEBenchmark:
    def __init__(self, task_name, seed=42):
        self.task_name = task_name
        self.seed = seed

    def __call__(self, model: BrainModel):
        model.mode = BrainModel.Modes.sentence_features
        data_dir = self.task_name.upper().replace('COLA', 'CoLA')
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
                                                'ressources', 'ml', 'glue', data_dir))
        set_seed(self.seed)
        max_seq_length = 128
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prepare GLUE task
        if self.task_name not in processors:
            raise ValueError("Task not found: %s" % self.task_name)
        processor = processors[self.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)

        decoder_head = DecoderHead(features_size=model.features_size, num_labels=num_labels)
        decoder_head = decoder_head.to(device)

        # setup Evaluation
        eval_task_names = ("mnli", "mnli-mm") if self.task_name == "mnli" else (self.task_name,)

        def run_evaluation(return_score=False):
            scores = []
            # Loop to handle MNLI double evaluation (matched, mis-matched)
            for eval_task in eval_task_names:
                examples, label_list, output_mode = get_examples(data_dir=data_dir, task=eval_task, evaluate=True)
                eval_dataset = model.glue_dataset(task=eval_task, examples=examples, label_list=label_list,
                                                  output_mode=output_mode, max_seq_length=max_seq_length)
                result = evaluate(features_model=model, decoder_head=decoder_head,
                                  eval_dataset=eval_dataset, task_name=eval_task, output_mode=output_mode,
                                  device=device)
                if not return_score:
                    return result  # we're ignoring mnli-mm here, but this return is just for progress logging anyway
                score = Score([[value for key, value in result.items()]],
                              coords={'eval_task': [eval_task], 'measure': list(result.keys())},
                              dims=['eval_task', 'measure'])
                score.attrs['data_dir'] = data_dir
                score.attrs['benchmark_identifier'] = f"glue-{self.task_name}"
                score.attrs['eval_task'] = eval_task
                score.attrs['model_identifier'] = model.identifier
                scores.append(score)
            scores = Score.merge(*scores)
            return scores

        # Training
        examples, label_list, output_mode = get_examples(data_dir=data_dir, task=self.task_name, evaluate=False)
        train_dataset = model.glue_dataset(task=self.task_name, examples=examples, label_list=label_list,
                                           output_mode=output_mode, max_seq_length=max_seq_length)
        train(features_model=model, decoder_head=decoder_head,
              train_dataset=train_dataset, run_evaluation=run_evaluation,
              seed=self.seed, device=device)

        # Evaluation
        logger.info("Evaluate")
        results = run_evaluation(return_score=True)
        return results


benchmark_pool = {f'glue-{task}': LazyLoad(lambda _task=task: GLUEBenchmark(_task))
                  for task in ['cola', 'mnli', 'mrpc', 'sst-2', 'sts-b', 'qqp', 'qnli', 'rte', 'wnli']}
