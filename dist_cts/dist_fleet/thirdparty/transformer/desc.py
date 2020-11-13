#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file demo1.py
  * @author liyang109@baidu.com
  * @date 2020-05-18 11:05
  * @brief
  *
  **************************************************************************/
"""
batch_size = -1
seq_len = 256
n_head = 8
d_model = 512

input_descs = {
    "src_word": [(batch_size, seq_len, 1), "int64", 2],
    "src_pos": [(batch_size, seq_len, 1), "int64"],
    "src_slf_attn_bias": [(batch_size, n_head, seq_len, seq_len), "float32"],
    "trg_word": [(batch_size, seq_len, 1), "int64", 2],
    "trg_pos": [(batch_size, seq_len, 1), "int64"],
    "trg_slf_attn_bias": [(batch_size, n_head, seq_len, seq_len), "float32"],
    "trg_src_attn_bias": [(batch_size, n_head, seq_len, seq_len), "float32"],
    "enc_output": [(batch_size, seq_len, d_model), "float32"],
    "lbl_word": [(batch_size * seq_len, 1), "int64"],
    "lbl_weight": [(batch_size * seq_len, 1), "float32"],
    "init_score": [(batch_size, 1), "float32", 2],
    "init_idx": [(batch_size, ), "int32"],
}

word_emb_param_names = (
    "src_word_emb_table",
    "trg_word_emb_table", )
pos_enc_param_names = (
    "src_pos_enc_table",
    "trg_pos_enc_table", )
encoder_data_input_fields = (
    "src_word",
    "src_pos",
    "src_slf_attn_bias", )
decoder_data_input_fields = (
    "trg_word",
    "trg_pos",
    "trg_slf_attn_bias",
    "trg_src_attn_bias",
    "enc_output", )
label_data_input_fields = (
    "lbl_word",
    "lbl_weight", )

fast_decoder_data_input_fields = (
    "trg_word",
    "init_score",
    "init_idx",
    "trg_src_attn_bias", )

dropout_seed = None
