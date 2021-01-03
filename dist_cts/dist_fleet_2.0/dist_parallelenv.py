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
  * @file test.py
  * @author liyang109@baidu.com
  * @date 2020-12-30 15:53
  * @brief
  *
  **************************************************************************/
"""
import paddle
import paddle.distributed as dist


def train():
    """parallelenv"""
    # 1. initialize parallel env
    dist.init_parallel_env()

    # 2. get current ParallelEnv
    parallel_env = dist.ParallelEnv()
    assert parallel_env.rank == 0
    assert parallel_env.world_size == 2
    print("test_ParallelEnv ... ok")


if __name__ == '__main__':
    train()