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
  * @file test_dist_fleet_static_ctr.py
  * @author liyang109@baidu.com
  * @date 2020-11-18 16:04
  * @brief 
  *
  **************************************************************************/
"""
from __future__ import print_function
import nose.tools as tools
from dist_base_fleet import TestFleetBase
import os
import json
from dist_base_fleet import run_by_freq
from dist_base_fleet import run_with_compatibility


class TestDistCTR(TestFleetBase):
    """Test dist ctr cases."""

    def __init__(self):
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.single_cpu_data = [
            2.2311227, 0.29134884, 0.18218634, 0.13182417, 0.1027305
        ]
        self.single_cpu_data_increment = [
            2.2311227, 0.29134884, 0.18218634, 0.13182417, 0.1027305
        ]
        self._model_file = 'dist_fleet_static_ctr.py'

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
            expect_data = self.single_cpu_data
        if delta:
            for i in range(len(expect_data)):
                tools.assert_almost_equal(loss[i], expect_data[i], delta=delta)
        else:
            for i in range(len(expect_data)):
                tools.assert_equal(loss[i], expect_data[i])

    """pyreader"""
    def test_ctr_1ps_1tr_pyreader_async(
            self):
        """test_ctr_1ps_1tr_pyreader_async."""
        TestFleetBase.__init__(self, pservers=1, trainers=1)
        self.run_params = {
            'mode': 'async',
            'reader': 'pyreader'
        }
        self.test_ctr_1ps_1tr_pyreader_async.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_1ps_1tr_pyreader_sync(
            self):
        """test_ctr_1ps_1tr_pyreader_sync."""
        TestFleetBase.__init__(self, pservers=1, trainers=1)
        self.run_params = {
            'mode': 'sync',
            'reader': 'pyreader'
        }
        self.test_ctr_1ps_1tr_pyreader_sync.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_1ps_1tr_pyreader_geo(
            self):
        """test_ctr_1ps_1tr_pyreader_geo."""
        TestFleetBase.__init__(self, pservers=1, trainers=1)
        self.run_params = {
            'mode': 'geo',
            'reader': 'pyreader'
        }
        self.test_ctr_1ps_1tr_pyreader_geo.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_1ps_1tr_pyreader_auto(
            self):
        """test_ctr_1ps_1tr_pyreader_auto."""
        TestFleetBase.__init__(self, pservers=1, trainers=1)
        self.run_params = {
            'mode': 'sync',
            'reader': 'pyreader'
        }
        self.test_ctr_1ps_1tr_pyreader_auto.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_1ps_2tr_pyreader_async(
            self):
        """test_ctr_1ps_2tr_pyreader_async."""
        TestFleetBase.__init__(self, pservers=1, trainers=2)
        self.run_params = {
            'mode': 'async',
            'reader': 'pyreader'
        }
        self.test_ctr_1ps_2tr_pyreader_async.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_1ps_2tr_pyreader_sync(
            self):
        """test_ctr_1ps_2tr_pyreader_sync."""
        TestFleetBase.__init__(self, pservers=1, trainers=2)
        self.run_params = {
            'mode': 'sync',
            'reader': 'pyreader'
        }
        self.test_ctr_1ps_2tr_pyreader_sync.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_1ps_2tr_pyreader_geo(
            self):
        """test_ctr_1ps_2tr_pyreader_geo."""
        TestFleetBase.__init__(self, pservers=1, trainers=2)
        self.run_params = {
            'mode': 'geo',
            'reader': 'pyreader'
        }
        self.test_ctr_1ps_2tr_pyreader_geo.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_1ps_2tr_pyreader_auto(
            self):
        """test_ctr_1ps_2tr_pyreader_auto."""
        TestFleetBase.__init__(self, pservers=1, trainers=2)
        self.run_params = {
            'mode': 'sync',
            'reader': 'pyreader'
        }
        self.test_ctr_1ps_2tr_pyreader_auto.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_1tr_pyreader_async(
            self):
        """test_ctr_2ps_1tr_pyreader_async."""
        TestFleetBase.__init__(self, pservers=2, trainers=1)
        self.run_params = {
            'mode': 'async',
            'reader': 'pyreader'
        }
        self.test_ctr_2ps_1tr_pyreader_async.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_1tr_pyreader_sync(
            self):
        """test_ctr_2ps_1tr_pyreader_sync."""
        TestFleetBase.__init__(self, pservers=2, trainers=1)
        self.run_params = {
            'mode': 'sync',
            'reader': 'pyreader'
        }
        self.test_ctr_2ps_1tr_pyreader_sync.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_1tr_pyreader_geo(
            self):
        """test_ctr_2ps_1tr_pyreader_geo."""
        TestFleetBase.__init__(self, pservers=2, trainers=1)
        self.run_params = {
            'mode': 'geo',
            'reader': 'pyreader'
        }
        self.test_ctr_2ps_1tr_pyreader_geo.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_1tr_pyreader_auto(
            self):
        """test_ctr_2ps_1tr_pyreader_auto."""
        TestFleetBase.__init__(self, pservers=2, trainers=1)
        self.run_params = {
            'mode': 'sync',
            'reader': 'pyreader'
        }
        self.test_ctr_2ps_1tr_pyreader_auto.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_pyreader_async(
            self):
        """test_ctr_2ps_2tr_pyreader_async."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'mode': 'async',
            'reader': 'pyreader'
        }
        self.test_ctr_2ps_2tr_pyreader_async.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_pyreader_sync(
            self):
        """test_ctr_2ps_2tr_pyreader_sync."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'mode': 'sync',
            'reader': 'pyreader'
        }
        self.test_ctr_2ps_2tr_pyreader_sync.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_pyreader_geo(
            self):
        """test_ctr_2ps_2tr_pyreader_geo."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'mode': 'geo',
            'reader': 'pyreader'
        }
        self.test_ctr_2ps_2tr_pyreader_geo.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_pyreader_auto(
            self):
        """test_ctr_2ps_2tr_pyreader_auto."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'mode': 'sync',
            'reader': 'pyreader'
        }
        self.test_ctr_2ps_2tr_pyreader_auto.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)
    """dataset"""
    def test_ctr_2ps_2tr_dataset_async(
            self):
        """test_ctr_2ps_2tr_dataset_async."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'mode': 'sync',
            'reader': 'pyreader'
        }
        self.test_ctr_2ps_2tr_dataset_async.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_1ps_2tr_dataset_async(
            self):
        """test_ctr_1ps_2tr_dataset_async."""
        TestFleetBase.__init__(self, pservers=1, trainers=2)
        self.run_params = {
            'mode': 'sync',
            'reader': 'pyreader'
        }
        self.test_ctr_1ps_2tr_dataset_async.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_1ps_1tr_dataset_async(
            self):
        """test_ctr_1ps_1tr_dataset_async."""
        TestFleetBase.__init__(self, pservers=1, trainers=1)
        self.run_params = {
            'mode': 'sync',
            'reader': 'pyreader'
        }
        self.test_ctr_1ps_1tr_dataset_async.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_1tr_dataset_async(
            self):
        """test_ctr_2ps_1tr_dataset_async."""
        TestFleetBase.__init__(self, pservers=2, trainers=1)
        self.run_params = {
            'mode': 'sync',
            'reader': 'pyreader'
        }
        self.test_ctr_2ps_1tr_dataset_async.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)
