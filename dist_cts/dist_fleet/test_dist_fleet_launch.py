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
  * @file test_dist_fleet_launch.py
  * @author liyang109@baidu.com
  * @date 2020-06-03 11:18
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
            "--selected_gpus=0,1 --log_dir=mylog --print_config=False --use_paddlecloud \
             --started_port=6070 --log_level=10 dist_fleet_launch_script.py",
            "--log_dir=mylog --print_config=True --use_paddlecloud \
             --started_port=6070 --log_level=10 dist_fleet_launch_script.py",
            "--selected_gpus=0,1 --log_dir=mylog --print_config=False --use_paddlecloud \
             --started_port=6070 --log_level=10 dist_fleet_launch_script.py",
            "--log_dir=mylog --print_config=False --use_paddlecloud \
             --started_port=6070 --log_level=10 dist_fleet_launch_script.py",
            "--selected_gpus=0,1 --log_dir=mylog --print_config=False --use_paddlecloud \
             --log_level=10 dist_fleet_launch_script.py",
            "--log_dir=mylog --print_config=True --use_paddlecloud \
             --log_level=10 dist_fleet_launch_script.py",
            "--selected_gpus=0,1 --log_dir=mylog --print_config=False --use_paddlecloud \
             --log_level=10 dist_fleet_launch_script.py",
            "--log_dir=mylog --print_config=False --use_paddlecloud \
             --log_level=10 dist_fleet_launch_script.py",
            "--selected_gpus=0,1 --log_dir=mylog --print_config=False --use_paddlecloud \
             --started_port=6070 dist_fleet_launch_script.py",
            "--log_dir=mylog --print_config=True --use_paddlecloud \
             --started_port=6070 dist_fleet_launch_script.py",
            "--selected_gpus=0,1 --log_dir=mylog --print_config=False --use_paddlecloud \
             --started_port=6070 dist_fleet_launch_script.py",
            "--log_dir=mylog --print_config=False --use_paddlecloud \
             --started_port=6070 dist_fleet_launch_script.py",
            "--selected_gpus=0,1 --log_dir=mylog --print_config=False --use_paddlecloud \
             dist_fleet_launch_script.py",
            "--log_dir=mylog --print_config=True --use_paddlecloud \
             dist_fleet_launch_script.py",
            "--selected_gpus=0,1 --log_dir=mylog --print_config=False --use_paddlecloud \
             dist_fleet_launch_script.py",
            "--log_dir=mylog --print_config=False --use_paddlecloud \
             dist_fleet_launch_script.py",
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
                "python -m paddle.distributed.launch --cluster_node_ips=127.0.0.1 --node_ip=127.0.0.1 " + cmd,
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

    def test_dist_launch_Tsg_Tld_Tpc_Tup_Tsp_Tll(self):
        """test_dist_launch_Tsg_Tld_Tpc_Tup_Tsp_Tll."""
        args = self.all_args[0]
        self.get_result(args)

    def test_dist_launch_Fsg_Tld_Tpc_Tup_Tsp_Tll(self):
        """test_dist_launch_Fsg_Tld_Tpc_Tup_Tsp_Tll."""
        args = self.all_args[1]
        self.get_result(args)

    def test_dist_launch_Tsg_Tld_Fpc_Tup_Tsp_Tll(self):
        """test_dist_launch_Tsg_Tld_Fpc_Tup_Tsp_Tll."""
        args = self.all_args[2]
        self.get_result(args)

    def test_dist_launch_Fsg_Tld_Fpc_Tup_Tsp_Tll(self):
        """test_dist_launch_Fsg_Tld_Fpc_Tup_Tsp_Tll."""
        args = self.all_args[3]
        self.get_result(args)

    def test_dist_launch_Tsg_Tld_Tpc_Tup_Fsp_Tll(self):
        """test_dist_launch_Tsg_Tld_Tpc_Tup_Fsp_Tll."""
        args = self.all_args[4]
        self.get_result(args)

    def test_dist_launch_Fsg_Tld_Tpc_Tup_Fsp_Tll(self):
        """test_dist_launch_Fsg_Tld_Tpc_Tup_Fsp_Tll."""
        args = self.all_args[5]
        self.get_result(args)

    def test_dist_launch_Tsg_Tld_Fpc_Tup_Fsp_Tll(self):
        """test_dist_launch_Tsg_Tld_Fpc_Tup_Fsp_Tll."""
        args = self.all_args[6]
        self.get_result(args)

    def test_dist_launch_Fsg_Tld_Fpc_Tup_Fsp_Tll(self):
        """test_dist_launch_Fsg_Tld_Fpc_Tup_Fsp_Tll."""
        args = self.all_args[7]
        self.get_result(args)
    
    def test_dist_launch_Tsg_Tld_Tpc_Tup_Tsp_Fll(self):
        """test_dist_launch_Tsg_Tld_Tpc_Tup_Tsp_Fll."""
        args = self.all_args[8]
        self.get_result(args)

    def test_dist_launch_Fsg_Tld_Tpc_Tup_Tsp_Fll(self):
        """test_dist_launch_Fsg_Tld_Tpc_Tup_Tsp_Fll."""
        args = self.all_args[9]
        self.get_result(args)

    def test_dist_launch_Tsg_Tld_Fpc_Tup_Tsp_Fll(self):
        """test_dist_launch_Tsg_Tld_Fpc_Tup_Tsp_Fll."""
        args = self.all_args[10]
        self.get_result(args)

    def test_dist_launch_Fsg_Tld_Fpc_Tup_Tsp_Fll(self):
        """test_dist_launch_Fsg_Tld_Fpc_Tup_Tsp_Fll."""
        args = self.all_args[11]
        self.get_result(args)

    def test_dist_launch_Tsg_Tld_Tpc_Tup_Fsp_Fll(self):
        """test_dist_launch_Tsg_Tld_Tpc_Tup_Fsp_Fll."""
        args = self.all_args[12]
        self.get_result(args)

    def test_dist_launch_Fsg_Tld_Tpc_Tup_Fsp_Fll(self):
        """test_dist_launch_Fsg_Tld_Tpc_Tup_Fsp_Fll."""
        args = self.all_args[13]
        self.get_result(args)

    def test_dist_launch_Tsg_Tld_Fpc_Tup_Fsp_Fll(self):
        """test_dist_launch_Tsg_Tld_Fpc_Tup_Fsp_Fll."""
        args = self.all_args[14]
        self.get_result(args)

    def test_dist_launch_Fsg_Tld_Fpc_Tup_Fsp_Fll(self):
        """test_dist_launch_Fsg_Tld_Fpc_Tup_Fsp_Fll."""
        args = self.all_args[15]
        self.get_result(args)
    
    