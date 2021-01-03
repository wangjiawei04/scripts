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
from paddle.distributed import init_parallel_env
import paddle


def barrier():
    """barrier"""
    paddle.set_device('gpu:%d' % paddle.distributed.ParallelEnv().dev_id)
    init_parallel_env()
    paddle.distributed.barrier()
    print("test_barrier ... ok")


if __name__ == '__main__':
    barrier()