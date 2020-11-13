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
  * @file test_dist_fleet_vgg.py
  * @author liyang109@baidu.com
  * @date 2020-11-05 15:53
  * @brief
  *
  **************************************************************************/
"""
from __future__ import print_function
import nose.tools as tools
import os
from dist_base_fleet import TestFleetBase
from dist_base_fleet import run_by_freq
import json


class TestDistVgg16(TestFleetBase):
    """VGG test cases."""
    def __init__(self):
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.single_sync_gpu_data = [
            1.5387154, 1.5342727, 1.525869, 1.5140724, 1.499335
        ]
        self._model_file = 'dist_fleet_vgg.py'

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
            expect_data = self.single_sync_gpu_data
        if delta:
            for i in range(len(expect_data)):
                tools.assert_almost_equal(loss[i], expect_data[i], delta=delta)
        else:
            for i in range(len(expect_data)):
                tools.assert_equal(loss[i], expect_data[i])

    """ FP32 """

    def test_1t_2g_Tdgc(self):
        """test_1t_2g_Tdgc."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'dgc': True,
            'lars': False,
            'lamb': False,
            'auto': False,
            'sync_bn': False,
            'nccl_comm_num': 1,
            'fuse_all_reduce_ops': False,
            'sync_nccl_allreduce': False,
            'inter_nranks': 8,
            'recompute': False,
            'pipeline': False,
            'localsgd': False,
            'gradient_merge': False,
            'fp16': False,
            'num_threads': 2,
            'enable_inplace': False,
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1t_2g_Tlars(self):
        """test_1t_2g_Tlars."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'dgc': False,
            'lars': True,
            'lamb': False,
            'auto': False,
            'sync_bn': False,
            'nccl_comm_num': 1,
            'fuse_all_reduce_ops': False,
            'sync_nccl_allreduce': False,
            'inter_nranks': 8,
            'recompute': False,
            'pipeline': False,
            'localsgd': False,
            'gradient_merge': False,
            'fp16': False,
            'num_threads': 2,
            'enable_inplace': False,
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1t_2g_Tlamb(self):
        """test_1t_2g_Tlamb."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'dgc': False,
            'lars': False,
            'lamb': True,
            'auto': False,
            'sync_bn': False,
            'nccl_comm_num': 1,
            'fuse_all_reduce_ops': False,
            'sync_nccl_allreduce': False,
            'inter_nranks': 8,
            'recompute': False,
            'pipeline': False,
            'localsgd': False,
            'gradient_merge': False,
            'fp16': False,
            'num_threads': 2,
            'enable_inplace': False,
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1t_2g_Tauto(self):
        """test_1t_2g_Tauto."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'dgc': False,
            'lars': False,
            'lamb': False,
            'auto': True,
            'sync_bn': False,
            'nccl_comm_num': 1,
            'fuse_all_reduce_ops': False,
            'sync_nccl_allreduce': False,
            'inter_nranks': 8,
            'recompute': False,
            'pipeline': False,
            'localsgd': False,
            'gradient_merge': False,
            'fp16': False,
            'num_threads': 2,
            'enable_inplace': False,
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1t_2g_Tsyncbn(self):
        """test_1t_2g_Tsyncbn."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'dgc': False,
            'lars': False,
            'lamb': False,
            'auto': False,
            'sync_bn': True,
            'nccl_comm_num': 1,
            'fuse_all_reduce_ops': False,
            'sync_nccl_allreduce': False,
            'inter_nranks': 8,
            'recompute': False,
            'pipeline': False,
            'localsgd': False,
            'gradient_merge': False,
            'fp16': False,
            'num_threads': 2,
            'enable_inplace': False,
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1t_2g_Tfuse_allreduceops(self):
        """test_1t_2g_Tfuse_allreduceops."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'dgc': False,
            'lars': False,
            'lamb': False,
            'auto': False,
            'sync_bn': False,
            'nccl_comm_num': 1,
            'fuse_all_reduce_ops': True,
            'sync_nccl_allreduce': False,
            'inter_nranks': 8,
            'recompute': False,
            'pipeline': False,
            'localsgd': False,
            'gradient_merge': False,
            'fp16': False,
            'num_threads': 2,
            'enable_inplace': False,
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1t_2g_Tsyncallreduce(self):
        """test_1t_2g_Tsyncallreduce."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'dgc': False,
            'lars': False,
            'lamb': False,
            'auto': False,
            'sync_bn': False,
            'nccl_comm_num': 1,
            'fuse_all_reduce_ops': False,
            'sync_nccl_allreduce': True,
            'inter_nranks': 8,
            'recompute': False,
            'pipeline': False,
            'localsgd': False,
            'gradient_merge': False,
            'fp16': False,
            'num_threads': 2,
            'enable_inplace': False,
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1t_2g_Trecompute(self):
        """test_1t_2g_Trecompute."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'dgc': False,
            'lars': False,
            'lamb': False,
            'auto': False,
            'sync_bn': False,
            'nccl_comm_num': 1,
            'fuse_all_reduce_ops': False,
            'sync_nccl_allreduce': False,
            'inter_nranks': 8,
            'recompute': True,
            'pipeline': False,
            'localsgd': False,
            'gradient_merge': False,
            'fp16': False,
            'num_threads': 2,
            'enable_inplace': False,
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    # has a bug, waitting for fix.
    # def test_1t_2g_Tpipeline(self):
    #     """test_1t_2g_Tpipeline."""
    #     TestFleetBase.__init__(self, pservers=0, trainers=1)
    #     self.run_params = {
    #         'dgc': False,
    #         'lars': False,
    #         'lamb': False,
    #         'auto': False,
    #         'sync_bn': False,
    #         'nccl_comm_num': 1,
    #         'fuse_all_reduce_ops': False,
    #         'sync_nccl_allreduce': False,
    #         'inter_nranks': 8,
    #         'recompute': False,
    #         'pipeline': True,
    #         'localsgd': False,
    #         'gradient_merge': False,
    #         'fp16': False,
    #         'num_threads': 2,
    #         'enable_inplace': False,
    #     }
    #     train_data_list1 = self.get_result(
    #         self._model_file, update_method='nccl', gpu_num=2)
    #     train_data_list2 = self.get_result(
    #         self._model_file, update_method='nccl', gpu_num=2)
    #     assert len(train_data_list1) == 2
    #     assert len(train_data_list2) == 2
    #     self.check_data(
    #         train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
    #     self.check_data(
    #         train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1t_2g_Tlsgd(self):
        """test_1t_2g_Tlsgd."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'dgc': False,
            'lars': False,
            'lamb': False,
            'auto': False,
            'sync_bn': False,
            'nccl_comm_num': 1,
            'fuse_all_reduce_ops': False,
            'sync_nccl_allreduce': False,
            'inter_nranks': 8,
            'recompute': False,
            'pipeline': False,
            'localsgd': True,
            'gradient_merge': False,
            'fp16': False,
            'num_threads': 2,
            'enable_inplace': False,
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1t_2g_Tgradmerge(self):
        """test_1t_2g_Tgradmerge."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'dgc': False,
            'lars': False,
            'lamb': False,
            'auto': False,
            'sync_bn': False,
            'nccl_comm_num': 1,
            'fuse_all_reduce_ops': False,
            'sync_nccl_allreduce': False,
            'inter_nranks': 8,
            'recompute': False,
            'pipeline': False,
            'localsgd': True,
            'gradient_merge': True,
            'fp16': False,
            'num_threads': 2,
            'enable_inplace': False,
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1t_2g_Tei(self):
        """test_1t_2g_Tei."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'dgc': False,
            'lars': False,
            'lamb': False,
            'auto': False,
            'sync_bn': False,
            'nccl_comm_num': 1,
            'fuse_all_reduce_ops': False,
            'sync_nccl_allreduce': False,
            'inter_nranks': 8,
            'recompute': False,
            'pipeline': False,
            'localsgd': False,
            'gradient_merge': False,
            'fp16': False,
            'num_threads': 2,
            'enable_inplace': True,
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1t_2g_Tdgc_Trecompute(self):
        """test_1t_2g_Tdgc_Trecompute."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'dgc': True,
            'lars': False,
            'lamb': False,
            'auto': False,
            'sync_bn': False,
            'nccl_comm_num': 1,
            'fuse_all_reduce_ops': False,
            'sync_nccl_allreduce': False,
            'inter_nranks': 8,
            'recompute': True,
            'pipeline': False,
            'localsgd': False,
            'gradient_merge': False,
            'fp16': False,
            'num_threads': 2,
            'enable_inplace': False,
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)