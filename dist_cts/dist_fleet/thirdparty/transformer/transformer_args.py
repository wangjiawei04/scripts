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
  * @file transformer_args.py
  * @author liyang109@baidu.com
  * @date 2020-05-17 16:29
  * @brief
  *
  **************************************************************************/
"""
import argparse
import ast
import reader
from config import *

DATA_DIR="./thirdparty/data/dist_data/transformer_data/"

def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser("Training for Transformer.")
    parser.add_argument(
        "--src_vocab_fpath",
        type=str,
        default=DATA_DIR + "/wmt16_ende_data_bpe/vocab_all.bpe.32000",
        required=False,
        help="The path of vocabulary file of source language.")
    parser.add_argument(
        "--trg_vocab_fpath",
        type=str,
        default=DATA_DIR + "/wmt16_ende_data_bpe/vocab_all.bpe.32000",
        required=False,
        help="The path of vocabulary file of target language.")
    parser.add_argument(
        "--train_file_pattern",
        type=str,
        required=False,
        default=DATA_DIR + "/wmt16_ende_data_bpe/train.tok.clean.bpe.32000.en-de",
        help="The pattern to match training data files.")
    parser.add_argument(
        "--num_trainers",
        type=int,
        default=1,
        help="num trainers")
    parser.add_argument(
        "--val_file_pattern",
        type=str,
        help="The pattern to match validation data files.")
    parser.add_argument(
        "--use_token_batch",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to "
        "produce batch data according to token number.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="The number of sequences contained in a mini-batch, or the maximum "
        "number of tokens (include paddings) contained in a mini-batch. Note "
        "that this represents the number on single device and the actual batch "
        "size for multi-devices will multiply the device number.")
    parser.add_argument(
        "--pool_size",
        type=int,
        default=200000,
        help="The buffer size to pool data.")
    parser.add_argument(
        "--sort_type",
        default="pool",
        choices=("global", "pool", "none"),
        help="The grain to sort by length: global for all instances; pool for "
        "instances in pool; none for no sort.")
    parser.add_argument(
        "--shuffle",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to shuffle instances in each pass.")
    parser.add_argument(
        "--shuffle_batch",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to shuffle the data batches.")
    parser.add_argument(
        "--special_token",
        type=str,
        default=["<s>", "<e>", "<unk>"],
        nargs=3,
        help="The <bos>, <eos> and <unk> tokens in the dictionary.")
    parser.add_argument(
        "--token_delimiter",
        type=lambda x: str(x.encode().decode("unicode-escape")),
        default=" ",
        help="The delimiter used to split tokens in source or target sentences. "
        "For EN-DE BPE data we provided, use spaces as token delimiter. ")
    parser.add_argument(
        'opts',
        help='See config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)
    parser.add_argument(
        '--sync', type=ast.literal_eval, default=True, help="sync mode.")
    parser.add_argument(
        "--use_py_reader",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to use py_reader.")
    parser.add_argument(
        "--use_fp16",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to use fp16.")
    parser.add_argument(
        "--run_benchmark",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to run as benchmark.")
    parser.add_argument(
        "--loss_scaling",
        type=float,
        default=64.0,
        help="The initial value for loss scaling.")
    parser.add_argument(
        "--fetch_steps",
        type=int,
        default=1,
        help="The frequency to fetch and print output.")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="How many epochs to run.")
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
    # Append args related to dict
    src_dict = reader.DataReader.load_dict(args.src_vocab_fpath)
    trg_dict = reader.DataReader.load_dict(args.trg_vocab_fpath)
    dict_args = [
        "src_vocab_size", str(len(src_dict)), "trg_vocab_size",
        str(len(trg_dict)), "bos_idx", str(src_dict[args.special_token[0]]),
        "eos_idx", str(src_dict[args.special_token[1]]), "unk_idx",
        str(src_dict[args.special_token[2]])
    ]
    merge_cfg_from_list(args.opts + dict_args,
                        [TrainTaskConfig, ModelHyperParams])

    return args
