#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
#======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
#======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file bert_args.py
  * @author liyang109@baidu.com
  * @date 2020-05-14 13:54
  * @brief 
  *
  **************************************************************************/
"""
import argparse
from utils.args import ArgumentGroup, print_arguments, check_cuda
DATA_DIR = './thirdparty/data/dist_data/bert_data/'


def p_args():
    parser = argparse.ArgumentParser(__doc__)
    model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
    model_g.add_arg("bert_config_path", str, None, "Path to the json file for bert model config.")
    model_g.add_arg("init_checkpoint", str, None, "Init checkpoint to resume training from.")
    model_g.add_arg("init_pretraining_params", str, DATA_DIR + u"ncased_L-24_H-1024_A-16/params",
                    "Init pre-training params which preforms fine-tuning from. If the "
                    "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")
    model_g.add_arg("checkpoints", str, "$PWD/tmp", "Path to save checkpoints.")
    train_g = ArgumentGroup(parser, "training", "training options.")
    train_g.add_arg("epoch", int, 3, "Number of epoches for fine-tuning.")
    train_g.add_arg("learning_rate", float, 5e-5, "Learning rate used to train with warmup.")
    train_g.add_arg("lr_scheduler", str, "linear_warmup_decay",
                    "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
    train_g.add_arg("weight_decay", float, 0.01, "Weight decay rate for L2 regularizer.")
    train_g.add_arg("warmup_proportion", float, 0.1,
                    "Proportion of training steps to perform linear learning rate warmup for.")
    train_g.add_arg("loss_scaling", float, 1.0,
                    "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")

    log_g = ArgumentGroup(parser, "logging", "logging related.")
    log_g.add_arg("skip_steps", int, 1, "The steps interval to print loss.")
    log_g.add_arg("verbose", bool, False, "Whether to output verbose log.")

    data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
    data_g.add_arg("data_dir", str, DATA_DIR + "xnli", "Path to training data.")
    data_g.add_arg("vocab_path", str, DATA_DIR + "uncased_L-24_H-1024_A-16/vocab.txt", "Vocabulary path.")
    data_g.add_arg("max_seq_len", int, 32, "Number of words of the longest seqence.")
    data_g.add_arg("batch_size", int, 5, "Total examples' number in batch for training. see also --in_tokens.")
    data_g.add_arg("in_tokens", bool, False,
                   "If set, the batch size will be the maximum number of tokens in one batch. "
                   "Otherwise, it will be the maximum number of examples in one batch.")
    data_g.add_arg("do_lower_case", bool, True,
                   "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
    data_g.add_arg("random_seed", int, 0, "Random seed.")
    data_g.add_arg("shuffle_seed", int, 2, "Shuffle seed.")

    run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
    run_type_g.add_arg("use_fast_executor", bool, False, "If set, use fast parallel executor (in experiment).")
    run_type_g.add_arg("shuffle", bool, False, "")
    run_type_g.add_arg("num_iteration_per_drop_scope", int, 1,
                       "Ihe iteration intervals to clean up temporary variables.")
    run_type_g.add_arg("task_name", str, "XNLI",
                       "The name of task to perform fine-tuning, should be in {'xnli', 'mnli', 'cola', 'mrpc'}.")
    run_type_g.add_arg("do_train", bool, True, "Whether to perform training.")
    parser.add_argument(
        '--update_method',
        type=str,
        required=True,
        choices=['pserver', 'nccl'])
    parser.add_argument(
        '--role', type=str, required=True, choices=['pserver', 'trainer'])
    parser.add_argument(
        '--endpoints', type=str, required=False, default="")
    parser.add_argument(
        '--current_id', type=int, required=False, default=0)
    parser.add_argument('--trainers', type=int, required=False, default=1)
    parser.add_argument(
        '--run_params', type=str, required=False, default='{}')
    args = parser.parse_args()

    return args