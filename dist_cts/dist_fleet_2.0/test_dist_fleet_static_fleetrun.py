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
  * @file test_dist_fleet_static_fleetrun.py
  * @author liyang109@baidu.com
  * @date 2020-11-17 17:32
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
class TestDistLaunch():
    """Test paddle.distributed.launch module cases."""
    def __init__(self):
        self.single_data = [0.70575, 0.69835, 0.69342, 0.690098, 0.687781]
        self.test_info1 = []
        self.test_info2 = []
        self.all_args = [
            "--gpus=0,1 --log_dir=mylog  dist_fleet_static_fleetrun.py",
            "--gpus=0,1  dist_fleet_static_fleetrun.py",
            "--log_dir=mylog  dist_fleet_static_fleetrun.py",
            "dist_fleet_static_fleetrun.py",
            "--gpus=0 --log_dir=mylog  dist_fleet_static_fleetrun.py",
            "--gpus=0 dist_fleet_static_fleetrun.py",
        ]
    def check_data(self, loss, delta=None, expect=None):
        """
        校验结果数据.
        Args:
            loss (list): the loss will be checked.
            delta (float):
            expect (list):
        """
        if expect:
            expect_data = expect
        else:
            expect_data = self.single_data
        if delta:
            for i in range(len(expect_data)):
                tools.assert_almost_equal(loss[i], expect_data[i], delta=delta)
        else:
            for i in range(len(expect_data)):
                tools.assert_equal(loss[i], expect_data[i])
    def start_proc(self, cmd):
        """start process."""
        p = subprocess.Popen(
            "fleetrun " + cmd,
            shell=True,
            stderr=open("/tmp/launch.log", "wb"),
            stdout=subprocess.PIPE)
        p.communicate()
        with open('mylog/workerlog.0', 'r') as f:
            lines = f.readlines()[-1].lstrip('[').rstrip(']\n').split(',')
        loss = [eval(i) for i in lines]
        return loss
    def get_result(self, args):
        """get result"""
        loss1 = self.start_proc(args)
        time.sleep(2)
        loss2 = self.start_proc(args)
        self.test_info1.append(loss1)
        self.test_info2.append(loss2)
        assert len(self.test_info1[0]) == 5
        assert len(self.test_info2[0]) == 5
        self.check_data(
            loss=self.test_info1[0], delta=3e-1, expect=self.test_info1[0])
        self.check_data(
            loss=self.test_info2[0], delta=3e-1, expect=self.single_data)
    def test_dist_launch_2gpus_ldir(self):
        """test_dist_launch_2gpus_ldir."""
        args = self.all_args[0]
        self.get_result(args)
    def test_dist_launch_2gpus(self):
        """test_dist_launch_2gpus."""
        args = self.all_args[1]
        self.get_result(args)
    def test_dist_launch_defaultgpus_ldir(self):
        """test_dist_launch_defaultgpus_ldir."""
        args = self.all_args[2]
        self.get_result(args)
    def test_dist_launch_defaultgpus(self):
        """test_dist_launch_defaultgpus."""
        args = self.all_args[3]
        self.get_result(args)
    def test_dist_launch_1gpus_ldir(self):
        """test_dist_launch_1gpus_ldir."""
        args = self.all_args[2]
        self.get_result(args)
    def test_dist_launch_1gpus(self):
        """test_dist_launch_1gpus."""
        args = self.all_args[3]
        self.get_result(args)
