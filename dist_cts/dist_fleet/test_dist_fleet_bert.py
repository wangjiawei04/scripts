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
  * @file dist_bert.py
  * @author liyang109@baidu.com
  * @date 2020-05-13 15:53
  * @brief
  *
  **************************************************************************/
"""
from __future__ import print_function
import nose.tools as tools
from dist_base_fleet import TestFleetBase


class TestDistBert(TestFleetBase):
    """BERT test cases."""

    def __init__(self):
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.single_sync_gpu_data = [1.16017, 1.051194, 0.0261433, 0.004292, 0.002873]
        self._model_file = 'dist_fleet_bert.py'

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

    def test_1tr_2gpu_nccl_fp32_Tei_nn2_Tlsgd_mc_cl(self):
        """test_1tr_2gpu_nccl_fp32_Tei_nn2_Tlsgd_mc_cl."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            #'fp16': False,
            'num_threads': 2,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': True,
            'mode': 'collective',
            'collective': "local_sgd"
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

    def test_1tr_2gpu_nccl_fp32_Fei_nn2_Tlsgd_mc_cl(self):
        """test_1tr_2gpu_nccl_fp32_Fei_nn2_Tlsgd_mc_cl."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            #'fp16': False,
            'num_threads': 2,
            'enable_inplace': False,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': True,
            'mode': 'collective',
            'collective': "local_sgd"
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

    def test_1tr_2gpu_nccl_fp32_Tei_nn2_Flsgd_mc_cl(self):
        """test_1tr_2gpu_nccl_fp32_Tei_nn2_Flsgd_mc_cl."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            #'fp16': False,
            'num_threads': 2,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': False,
            'mode': 'collective',
            'collective': "local_sgd"
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

    def test_1tr_2gpu_nccl_fp32_Fei_nn2_Flsgd_mc_cl(self):
        """test_1tr_2gpu_nccl_fp32_Fei_nn2_Flsgd_mc_cl."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            #'fp16': False,
            'num_threads': 2,
            'enable_inplace': False,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': False,
            'mode': 'collective',
            'collective': "local_sgd"
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

    def test_1tr_2gpu_nccl_fp32_Tei_nn2_Flsgd_mn_cn(self):
        """test_1tr_2gpu_nccl_fp32_Tei_nn2_Flsgd_mn_cn."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            #'fp16': False,
            'num_threads': 2,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': False,
            'mode': 'nccl2',
            'collective': "None"
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

    def test_1tr_2gpu_nccl_fp32_Fei_nn2_Flsgd_mn_cn(self):
        """test_1tr_2gpu_nccl_fp32_Fei_nn2_Flsgd_mn_cn."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            #'fp16': False,
            'num_threads': 2,
            'enable_inplace': False,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': False,
            'mode': 'nccl2',
            'collective': "None"
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


