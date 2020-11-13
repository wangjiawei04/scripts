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
  * @file test_dist_fleet_flags.py
  * @author liyang109@baidu.com
  * @date 2019-12-27 14:24
  * @brief 
  *
  **************************************************************************/
"""
from __future__ import print_function
import nose.tools as tools
from dist_base_fleet import TestFleetBase
import json
import os
from dist_base_fleet import run_by_freq


class TestDistSimnet(TestFleetBase):
    def __init__(self):
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.single_cpu_data = [0.094925374, 0.08564686, 0.0873276, 0.089603186, 0.0914526]
        self._model_file = 'dist_fleet_simnet_bow.py'

    def check_data(self, loss, delta=None, expect=None):
        """
        校验结果数据
        """
        if expect:
            expect_data = expect
        else:
            expect_data = self.single_cpu_data
        if delta:
            for i in range(len(expect_data)):
                tools.assert_almost_equal(loss[i], expect_data[i], delta=delta)
        else:
            for i in range(len(expect_data)):
                tools.assert_equal(loss[i], expect_data[i])

    """async"""

    def test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Tir_Tfr_Tms_Tdr(self):
        """
        test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Tir_Tfr_Tms_Tdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 1,'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 1, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 1,
                            'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Tir_Tfr_Tms_Tdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Tdr(self):
        """
        test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Tdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 0,'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 1, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 1,
                            'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Tdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Tdr(self):
        """
        test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Tdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 1, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 0, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 1,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Tdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Tdr(self):
        """
        test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Tdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 0, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 0, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 1,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Tdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Fdr(self):
        """
        test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Fdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 0,'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 1, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 0,
                            'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Fdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Fdr(self):
        """
        test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Fdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 1, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 0, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 0,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Fdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Fdr(self):
        """
        test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Fdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 0, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 0, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 0,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Fdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Tir_Tfr_Tms_Tdr(self):
        """
        test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Tir_Tfr_Tms_Tdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 1, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 1, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 1,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Tir_Tfr_Tms_Tdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Tdr(self):
        """
        test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Tdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 0, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 1, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 1,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Tdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Tdr(self):
        """
        test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Tdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 1, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 0, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 1,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Tdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Tdr(self):
        """
        test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Tdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 0, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 0, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 1,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Tdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Fdr(self):
        """
        test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Fdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 0, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 1, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 0,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Fdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Fdr(self):
        """
        test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Fdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 1, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 0, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 0,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Fdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Fdr(self):
        """
        test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Fdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 0, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 0, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 0,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Tgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Fdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Tir_Tfr_Tms_Tdr(self):
        """
        test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Tir_Tfr_Tms_Tdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 1, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 1, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 1,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Tir_Tfr_Tms_Tdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Tdr(self):
        """
        test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Tdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 0, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 1, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 1,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Tdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Tdr(self):
        """
        test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Tdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 1, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 0, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 1,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Tdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Tdr(self):
        """
        test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Tdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 0, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 0, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 1,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Tdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Fdr(self):
        """
        test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Fdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 0, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 1, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 0,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Fdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Fdr(self):
        """
        test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Fdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 1, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 0, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 0,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Fdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Fdr(self):
        """
        test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Fdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 0, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 0, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 0,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Fdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Tir_Tfr_Tms_Tdr(self):
        """
        test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Tir_Tfr_Tms_Tdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 1, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 1, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 1,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Tir_Tfr_Tms_Tdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Tdr(self):
        """
        test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Tdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 0, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 1, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 1,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Tdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Tdr(self):
        """
        test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Tdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 1, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 0, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 1,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Tdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Tdr(self):
        """
        test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Tdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 0, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 0, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 1,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Tdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Fdr(self):
        """
        test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Fdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 0, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 1, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 0,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Tms_Fdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Fdr(self):
        """
        test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Fdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 1, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 0, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 0,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Tir_Tfr_Fms_Fdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Fdr(self):
        """
        test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Fdr
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': False, 'async_mode': True, 'half_mode': False, 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 50, 'F_max_merge': 15,
                           'F_indept_recv': 0, 'F_fake_rpc': 1, 'F_g_thr_num': 5, 'F_s_thr_num': 5,
                           'F_mer_sparse': 0, 'F_min_send_grad': 20, 'F_queue_size': 20, 'F_s_wait_t': 5,
                           'F_t_pool_size': 10, 'F_thread_pool': 5, 'F_rpc_deadline': 5000, 'F_dis_reuse': 0,
                           'F_profile_path': os.getenv("LOG_PATH")}
        self.test_flag_2ps_2tr_async_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25_Fir_Tfr_Fms_Fdr.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)