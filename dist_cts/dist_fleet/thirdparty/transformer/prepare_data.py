#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file prepare_data.py
  * @author liyang109@baidu.com
  * @date 2020-05-17 18:21
  * @brief 
  *
  **************************************************************************/
"""
import numpy as np
import reader
from desc import *
import paddle.fluid as fluid
from config import *
from transformer_args import parse_args
from model import position_encoding_init

args = parse_args()
def pad_batch_data(insts,
                   pad_idx,
                   n_head,
                   is_target=False,
                   is_label=False,
                   return_attn_bias=True,
                   return_max_len=True,
                   return_num_token=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.
    inst_data = np.array(
        [inst + [pad_idx] * (max_len - len(inst)) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, 1])]
    if is_label:  # label weight
        inst_weight = np.array(
            [[1.] * len(inst) + [0.] * (max_len - len(inst)) for inst in insts])
        return_list += [inst_weight.astype("float32").reshape([-1, 1])]
    else:  # position data
        inst_pos = np.array([
            list(range(0, len(inst))) + [0] * (max_len - len(inst))
            for inst in insts
        ])
        return_list += [inst_pos.astype("int64").reshape([-1, 1])]
    if return_attn_bias:
        if is_target:
            # This is used to avoid attention on paddings and subsequent
            # words.
            slf_attn_bias_data = np.ones((inst_data.shape[0], max_len, max_len))
            slf_attn_bias_data = np.triu(slf_attn_bias_data,
                                         1).reshape([-1, 1, max_len, max_len])
            slf_attn_bias_data = np.tile(slf_attn_bias_data,
                                         [1, n_head, 1, 1]) * [-1e9]
        else:
            # This is used to avoid attention on paddings.
            slf_attn_bias_data = np.array([[0] * len(inst) + [-1e9] *
                                           (max_len - len(inst))
                                           for inst in insts])
            slf_attn_bias_data = np.tile(
                slf_attn_bias_data.reshape([-1, 1, 1, max_len]),
                [1, n_head, max_len, 1])
        return_list += [slf_attn_bias_data.astype("float32")]
    if return_max_len:
        return_list += [max_len]
    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]
    return return_list if len(return_list) > 1 else return_list[0]


def prepare_batch_input(insts, data_input_names, src_pad_idx, trg_pad_idx,
                        n_head, d_model):
    """
    Put all padded data needed by training into a dict.
    """
    src_word, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, n_head, is_target=False)
    src_word = src_word.reshape(-1, src_max_len, 1)
    src_pos = src_pos.reshape(-1, src_max_len, 1)
    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = pad_batch_data(
        [inst[1] for inst in insts], trg_pad_idx, n_head, is_target=True)
    trg_word = trg_word.reshape(-1, trg_max_len, 1)
    trg_pos = trg_pos.reshape(-1, trg_max_len, 1)

    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, trg_max_len, 1]).astype("float32")

    lbl_word, lbl_weight, num_token = pad_batch_data(
        [inst[2] for inst in insts],
        trg_pad_idx,
        n_head,
        is_target=False,
        is_label=True,
        return_attn_bias=False,
        return_max_len=False,
        return_num_token=True)

    data_input_dict = dict(
        zip(data_input_names, [
            src_word, src_pos, src_slf_attn_bias, trg_word, trg_pos,
            trg_slf_attn_bias, trg_src_attn_bias, lbl_word, lbl_weight
        ]))

    return data_input_dict, np.asarray([num_token], dtype="float32")


def prepare_data_generator(args,
                           is_test,
                           count,
                           pyreader,
                           py_reader_provider_wrapper,
                           place=None):
    """
    Data generator wrapper for DataReader. If use py_reader, set the data
    provider for py_reader
    """
    data_reader = reader.DataReader(
        fpattern=args.val_file_pattern if is_test else args.train_file_pattern,
        src_vocab_fpath=args.src_vocab_fpath,
        trg_vocab_fpath=args.trg_vocab_fpath,
        token_delimiter=args.token_delimiter,
        use_token_batch=args.use_token_batch,
        batch_size=args.batch_size * (1 if args.use_token_batch else count),
        pool_size=args.pool_size,
        sort_type=args.sort_type,
        shuffle=args.shuffle,
        shuffle_batch=args.shuffle_batch,
        start_mark=args.special_token[0],
        end_mark=args.special_token[1],
        unk_mark=args.special_token[2],
        # count start and end tokens out
        max_length=ModelHyperParams.max_length - 2,
        clip_last_batch=False).batch_generator

    def stack(data_reader, count, clip_last=True):
        """stack"""
        def __impl__():
            """impl"""
            res = []
            for item in data_reader():
                res.append(item)
                if len(res) == count:
                    yield res
                    res = []
            if len(res) == count:
                yield res
            elif not clip_last:
                data = []
                for item in res:
                    data += item
                if len(data) > count:
                    inst_num_per_part = len(data) // count
                    yield [
                        data[inst_num_per_part * i:inst_num_per_part * (i + 1)]
                        for i in range(count)
                    ]

        return __impl__

    def split(data_reader, count):
        """split method"""
        def __impl__():
            """impl"""
            for item in data_reader():
                inst_num_per_part = len(item) // count
                for i in range(count):
                    yield item[inst_num_per_part * i:inst_num_per_part * (i + 1
                                                                          )]

        return __impl__

    if not args.use_token_batch:
        # to make data on each device have similar token number
        data_reader = split(data_reader, count)
    if args.use_py_reader:
        if args.num_trainers > 1:
            data_reader = fluid.contrib.reader.distributed_batch_reader(data_reader)
        pyreader.decorate_tensor_provider(
            py_reader_provider_wrapper(data_reader, place))
        data_reader = None
    else:  # Data generator for multi-devices
        data_reader = stack(data_reader, count)
    return data_reader


def prepare_feed_dict_list(data_generator, init_flag, count):
    """
    Prepare the list of feed dict for multi-devices.
    """
    feed_dict_list = []
    if data_generator is not None:  # use_py_reader == False
        data_input_names = encoder_data_input_fields + \
                    decoder_data_input_fields[:-1] + label_data_input_fields
        data = next(data_generator)
        for idx, data_buffer in enumerate(data):
            data_input_dict, num_token = prepare_batch_input(
                data_buffer, data_input_names, ModelHyperParams.eos_idx,
                ModelHyperParams.eos_idx, ModelHyperParams.n_head,
                ModelHyperParams.d_model)
            feed_dict_list.append(data_input_dict)
    if init_flag:
        for idx in range(count):
            pos_enc_tables = dict()
            for pos_enc_param_name in pos_enc_param_names:
                pos_enc_tables[pos_enc_param_name] = position_encoding_init(
                    ModelHyperParams.max_length + 1, ModelHyperParams.d_model)
            if len(feed_dict_list) <= idx:
                feed_dict_list.append(pos_enc_tables)
            else:
                feed_dict_list[idx] = dict(
                    list(pos_enc_tables.items()) + list(feed_dict_list[idx]
                                                        .items()))

    return feed_dict_list if len(feed_dict_list) == count else None


def py_reader_provider_wrapper(data_reader, place):
    """
    Data provider needed by fluid.layers.py_reader.
    """

    def py_reader_provider():
        """reader provider"""
        data_input_names = encoder_data_input_fields + \
                    decoder_data_input_fields[:-1] + label_data_input_fields
        for batch_id, data in enumerate(data_reader()):
            data_input_dict, num_token = prepare_batch_input(
                data, data_input_names, ModelHyperParams.eos_idx,
                ModelHyperParams.eos_idx, ModelHyperParams.n_head,
                ModelHyperParams.d_model)
            total_dict = dict(data_input_dict.items())
            yield [total_dict[item] for item in data_input_names]

    return py_reader_provider



