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
class TrainTaskConfig(object):
    """
    TrainTaskConfig
    """
    use_gpu = True
    pass_num = 30
    learning_rate = 2.0
    beta1 = 0.9
    beta2 = 0.997
    eps = 1e-9
    warmup_steps = 8000
    label_smooth_eps = 0.1
    model_dir = "trained_models"
    ckpt_dir = "trained_ckpts"
    ckpt_path = None
    start_step = 0
    save_freq = 1000000


class InferTaskConfig(object):
    """
    InferTaskConfig
    """
    use_gpu = True
    batch_size = 10
    beam_size = 5
    max_out_len = 256
    n_best = 1
    output_bos = False
    output_eos = False
    output_unk = True
    model_path = "trained_models/pass_1.infer.model"


class ModelHyperParams(object):
    """
    ModelHyperParams
    """
    src_vocab_size = 10000
    trg_vocab_size = 10000
    bos_idx = 0
    eos_idx = 1
    unk_idx = 2
    max_length = 256
    d_model = 512
    d_inner_hid = 2048
    d_key = 64
    d_value = 64
    n_head = 8
    n_layer = 6
    prepostprocess_dropout = 0.3
    attention_dropout = 0.1
    relu_dropout = 0.1
    preprocess_cmd = "n"
    postprocess_cmd = "da"
    dropout_seed = None
    weight_sharing = True


def merge_cfg_from_list(cfg_list, g_cfgs):
    """
    Set the above global configurations using the cfg_list. 
    """
    assert len(cfg_list) % 2 == 0
    for key, value in zip(cfg_list[0::2], cfg_list[1::2]):
        for g_cfg in g_cfgs:
            if hasattr(g_cfg, key):
                try:
                    value = eval(value)
                except Exception:
                    pass
                setattr(g_cfg, key, value)
                break
