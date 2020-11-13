#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2019-12-25 19:48
# @Author  : liyang109
from __future__ import print_function
import nose.tools as tools
import time
import signal
import os
import subprocess


class TestDistPslib():
    def __init__(self):
        self.single_cpu_data = [0.6019563]

    def check_data(self, loss, delta=None, expect=None):
        """
        校验结果数据
        """
        if expect:
            expect_data = expect
        else:
            expect_data = self.single_cpu_data
        if delta:
            tools.assert_almost_equal(loss, expect_data, delta=delta)
        else:
            tools.assert_equal(loss, expect_data)

    def test_pslib(self):
        """test pslib"""
        test_info1 = []
        test_info2 = []

        p1 = subprocess.Popen(
            "mpirun --npernode 2 python/bin/python dist_fleet_pslib.py",
            shell=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        out, err =p1.communicate()
        lines = out.split("\n")[0]
        loss = [eval(lines)]
        test_info1.append(loss[0])
        print(test_info1)

        p2 = subprocess.Popen(
            "mpirun --npernode 2 python/bin/python dist_fleet_pslib.py",
            shell=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        out, err = p2.communicate()
        lines = out.split("\n")[0]
        loss = [eval(lines)]
        test_info2.append(loss[0])
        print(test_info2)

        assert len(test_info1[0]) == 1
        assert len(test_info2[0]) == 1
        self.check_data(loss=test_info1[0][0], expect=test_info2[0][0], delta=1e-0)
        self.check_data(loss=test_info1[0][0], expect=self.single_cpu_data[0], delta=1e-0)