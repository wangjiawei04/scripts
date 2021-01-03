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
import os
import subprocess


os.system("export CUDA_VISIBLE_DEVICES=0,1")
class TestApi():
    """test all api"""
    def test_all_gather(self):
        """test_all_gather"""
        cmd = 'fleetrun dist_all_gather.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0


    def test_all_reduce(self):
        """test_all_reduce"""
        cmd = 'fleetrun dist_all_reduce.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0

    def test_barrier(self):
        """test_barrier"""
        cmd = 'fleetrun dist_barrier.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0

    def test_broadcast(self):
        """test_broadcast"""
        cmd = 'fleetrun dist_broadcast.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0


    def test_get_rank(self):
        """test_get_rank"""
        cmd = 'fleetrun dist_get_rank.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0

    def test_get_world_size(self):
        """test_get_world_size"""
        cmd = 'fleetrun dist_get_world_size.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0

    def test_init_parallel_env(self):
        """test_init_parallel_env"""
        cmd = 'fleetrun dist_init_parallel_env.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0

    def test_parallelenv(self):
        """test_parallelenv"""
        cmd = 'fleetrun dist_parallelenv.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0

    def test_reduce(self):
        """test_reduce"""
        cmd = 'fleetrun dist_reduce.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0

    def test_reduceop(self):
        """test_reduceop"""
        cmd = 'fleetrun dist_reduceop.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0

    def test_scatter(self):
        """test_scatter"""
        cmd = 'fleetrun dist_scatter.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0