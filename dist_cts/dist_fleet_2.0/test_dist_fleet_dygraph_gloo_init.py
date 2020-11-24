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
  * @file dist_fleet_dygraph_gloo_init.py
  * @author liyang109@baidu.com
  * @date 2020-11-18 10:51
  * @brief 
  *
  **************************************************************************/
"""
import numpy as np
import paddle
import paddle.fluid as fluid
import sys
from paddle.fluid.dygraph.parallel import ParallelEnv
import subprocess
def dygraph_gloo_init():
    """test gloo init and broadcast"""
    paddle.distributed.init_parallel_env()
    if ParallelEnv().local_rank == 0:
        np_data = np.array([4, 5])
    else:
        np_data = np.array([1, 2])
    data = paddle.to_tensor(np_data)
    paddle.distributed.broadcast(data, 1)
    res = data.numpy()
    assert res == [1, 2]
if __name__ == "__main__":
    dygraph_gloo_init()
