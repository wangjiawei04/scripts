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
  * @file test_dist_fleet_dygraph.py
  * @author liyang109@baidu.com
  * @date 2020-11-16 14:40
  * @brief 
  *
  **************************************************************************/
"""
from __future__ import print_function
import nose.tools as tools
import time
import signal
import os
import subprocess
import time
class TestDygraph():
    """Test dygraph"""
    def __init__(self):
        self.single_data = []

    def test_dist_fleet_dygraph_api_2gpus(self):
      """test_dist_fleet_dygraph_api_2gpus."""
      cmd='fleetrun --gpu=0,1 dist_fleet_dygraph_api.py'
      pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      pro.wait()
      pro.returncode == 0

    def test_dist_fleet_dygraph_api_1gpus(self):
      """test_dist_fleet_dygraph_api_1gpus"""
      cmd='fleetrun --gpu=0 dist_fleet_dygraph_api.py'
      pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      pro.wait()
      pro.returncode == 0

    def test_dist_fleet_dygraph_lr_2gpus(self):
      """test_dist_fleet_dygraph_lr_2gpus"""
      cmd='fleetrun --gpu=0,1 dist_fleet_dygraph_lr.py'
      pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      pro.wait()
      pro.returncode == 0

    def test_dist_fleet_dygraph_lr_1gpus(self):
      """test_dist_fleet_dygraph_lr_1gpus"""
      cmd='fleetrun --gpu=0 dist_fleet_dygraph_lr.py'
      pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      pro.wait()
      pro.returncode == 0
