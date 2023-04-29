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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function
from collections import OrderedDict
import argparse
import csv
import logging
import os
import random
import sys
import io
import json
import shutil
import wandb

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from processors import QqpProcessor
from metrics_utils import compute_metrics

from transformers import AutoModelForSequenceClassification,AutoTokenizer
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from tensorboardX import SummaryWriter
logger = logging.getLogger(__name__)

    
def save_model_details(output_dir,model_to_save,tokenizer,result,tbwriter,args,log_step):
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)

    output_eval_metrics = os.path.join(output_dir, "eval_metrics.txt")
    with open(output_eval_metrics, "a") as writer:
        logger.info("***** Eval results {}*****".format(args.test_set))
        writer.write("***** Eval results {}*****\n".format(args.test_set))
        for key in sorted(result.keys()):
            if result[key] is not None and tbwriter is not None:
                tbwriter.add_scalar('{}/{}'.format(args.test_set, key), result[key], log_step)
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

def initialise_wandb(config):
    wandb.login()
    _ = wandb.init(
        # Set the project where this run will be logged
        project="nlp_project",
        # Track hyperparameters and run metadata
        config=config
    )


def main():
    os.environ['WANDB_MODE']='dryrun'
    logger.info("Running %s" % ' '.join(sys.argv))

    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--scan",
                        default="horizontal",
                        choices=["vertical", "horizontal"],
                        type=str,
                        help="The direction of linearizing table cells.")
    parser.add_argument("--output_dir",
                        default="outputs",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_dir",
                        type=str,
                        help="The output directory where the model checkpoints will be loaded during evaluation")
    parser.add_argument('--load_step',
                        type=int,
                        default=0,
                        help="The checkpoint step to be loaded")
    parser.add_argument("--fact",
                        default="first",
                        choices=["first", "second"],
                        type=str,
                        help="Whether to put fact in front.")
    parser.add_argument("--test_set",
                        default="dev",
                        choices=["dev", "test", "simple_test", "complex_test", "small_test"],
                        help="Which test set is used for evaluation",
                        type=str)
    parser.add_argument("--eval_batch_size",
                        default=6,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--balance",
                        action='store_true',
                        help="balance between + and - samples for training.")
    ## Other parameters
    parser.add_argument("--bert_model",
                        default="bert-base-multilingual-cased",
                        type=str,
                        help="bert-base-multilingual-cased, xlm-roberta-base")
    parser.add_argument("--task_name",
                        default="QQP",
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument('--period',
                        type=int,
                        default=500)
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=6,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=2,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--save_limit',
                        type=int, default=3,
                        help="How many previous models do you want to save?\n"
                        "Older models will be deleted to save memory\n")
    parser.add_argument("--input_save_dir",type=str)
    args = parser.parse_args()
    INPUT_SAVE_DIR=args.input_save_dir

    sys.stdout.flush()

    processors = {
        "qqp": QqpProcessor,
    }

    output_modes = {
        "qqp": "classification",
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.output_dir = "{}_fact-{}_{}-{}".format(args.output_dir, args.fact, args.scan,args.bert_model)
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    writer=SummaryWriter()

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    if args.load_dir:
        load_dir = args.load_dir
    else:
        load_dir = args.bert_model

    model = AutoModelForSequenceClassification.from_pretrained(load_dir,
                                                          cache_dir=cache_dir,
                                                          num_labels=num_labels)
    config_to_save=vars(args)
    config_to_save.update(model.config.to_dict())
    initialise_wandb(config_to_save)

    if args.fp16:
        model.half()
    model.to(device)
    if n_gpu>1:
        model=torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        
        optimizer = BertAdam(optimizer_grouped_parameters,
                                lr=args.learning_rate)

    global_step = 0
    tr_loss = 0
    num_saved=0
    last_deleted=0
    best_val_acc=0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids=torch.load(os.path.join(INPUT_SAVE_DIR, 'train_input_ids.pt'))
        all_input_mask=torch.load(os.path.join(INPUT_SAVE_DIR, 'train_attention_masks.pt'))
        all_label_ids=torch.load(os.path.join(INPUT_SAVE_DIR, 'train_labels.pt'))

        train_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            logger.info("Training epoch {} ...".format(epoch))
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration",position=0,leave=True)):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # define a new function to compute loss values for both output_modes
                logits = model.module(input_ids, segment_ids, input_mask, labels=None)

                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    total_norm = 0.0
                    for n, p in model.named_parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    preds = torch.argmax(logits, -1) == label_ids
                    acc = torch.sum(preds).float() / preds.size(0)

                    optimizer.step()
                    optimizer.zero_grad()
                    model.zero_grad()
                    global_step += 1

                if (step + 1) % args.period == 0:
                    # Save a trained model, configuration and tokenizer
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Only save the model it-self

                    # If we save using the predefined names, we can load using `from_pretrained`
                    output_dir = os.path.join(args.output_dir, 'save_step_{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    model.eval()
                    torch.set_grad_enabled(False)  # turn off gradient tracking
                    val_result=evaluate(args, model, device, processor, label_list, num_labels, tokenizer, output_mode, tr_loss,
                             global_step, task_name, tbwriter=writer, save_dir=output_dir)
                    
                    wandb.log(val_result)
                    
                    save_model_details(output_dir,model_to_save,tokenizer,val_result,args,global_step,tbwriter=writer)

                    if val_result['acc']>=best_val_acc:
                        output_dir = os.path.join(args.output_dir, 'best_model')
                        save_model_details(output_dir,model_to_save,tokenizer,val_result,args,global_step,tbwriter=writer)

                    num_saved+=1

                    if num_saved>args.save_limit:
                        last_deleted+=args.period
                        delete_path=os.path.join(args.output_dir, 'save_step_{}'.format(last_deleted))
                        shutil.rmtree(delete_path)

                    model.train()  # turn on train mode
                    torch.set_grad_enabled(True)  # start gradient tracking
                    tr_loss = 0

    # do eval before exit
    if args.do_eval:
        if not args.do_train:
            global_step = 0
            output_dir = None
        save_dir = output_dir if output_dir is not None else args.load_dir
        tbwriter = SummaryWriter(os.path.join(save_dir, 'eval/events'))
        load_step = args.load_step
        if args.load_dir is not None:
            load_step = int(os.path.split(args.load_dir)[1].replace('save_step_', ''))
            print("load_step = {}".format(load_step))
        model.eval()
        evaluate(args, model, device, processor, label_list, num_labels, tokenizer, output_mode, tr_loss,
                 global_step, task_name, tbwriter=tbwriter, save_dir=save_dir, load_step=load_step)


def evaluate(args, model, device, processor, label_list, num_labels, tokenizer, output_mode, tr_loss, global_step,
             task_name, tbwriter=None, save_dir=None, load_step=0):

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        all_input_ids=torch.load(os.path.join(args.output_dir, 'val_input_ids.pt'))
        all_input_mask=torch.load(os.path.join(args.output_dir, 'val_attention_masks.pt'))
        all_label_ids=torch.load(os.path.join(args.output_dir, 'val_labels.pt'))
        
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(all_input_ids))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        batch_idx = 0
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        temp = []

        for input_ids, input_mask, label_ids in tqdm(eval_dataloader, desc="Evaluating",position=0,leave=True):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model.module(input_ids, input_mask, labels=None)

            # create eval loss and other metric required by the task
            if output_mode == "classification":
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            elif output_mode == "regression":
                loss_fct = MSELoss()
                tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

            labels = label_ids.detach().cpu().numpy().tolist()
            input_string = tokenizer.decode(input_ids)
            facts=[ text.split('.')[0] for text in input_string ]
            labels = label_ids.detach().cpu().numpy().tolist()
            assert len(facts) == len(labels)

            temp.extend([(x, y, z) for x, y, z in zip(facts, labels)])
            batch_idx += 1

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)

        evaluation_results = OrderedDict()
        for x, y in zip(temp, preds):
            f, l = x
            if not f in evaluation_results:
                evaluation_results[f] = [{'fact': f, 'gold': int(l), 'pred': int(y)}]
            else:
                evaluation_results[f].append({'fact': f, 'gold': int(l), 'pred': int(y)})

        print("save_dir is {}".format(save_dir))
        output_eval_file = os.path.join(save_dir, "{}_eval_results.json".format(args.test_set))
        with io.open(output_eval_file, "w", encoding='utf-8 ') as fout:
            json.dump(evaluation_results, fout, sort_keys=True, indent=4)

        result = compute_metrics(task_name, preds, all_label_ids.numpy())
        loss = tr_loss/args.period if args.do_train and global_step > 0 else None

        log_step = global_step if args.do_train and global_step > 0 else load_step
        result['eval_loss'] = eval_loss
        result['global_step'] = log_step
        result['train_loss'] = loss

        logger.info("***** Eval results {}*****".format(args.test_set))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        return result


if __name__ == "__main__":
    main()
