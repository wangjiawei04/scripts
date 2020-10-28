#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(__file__))
from paddle_rec_base import RankDNNBaseNewConfig
import built_in
import sys

class TestUserDefine(RankDNNBaseNewConfig):
    """test user define model"""

    """test cpu cases"""
    def test_user_define_all_normal(self):
        """test normal yaml construct by MultiTaskMMOE base."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["runner"][0]["instance_class_path"] = 'paddle_rec_user_define.py'
        self.yaml_content["runner"][0]["network_class_path"] = 'paddle_rec_user_define.py'
        self.yaml_content["runner"][0]["startup_class_path"] = 'paddle_rec_user_define.py'
        self.yaml_content["runner"][0]["runner_class_path"] = 'paddle_rec_user_define.py'
        self.yaml_content["runner"][0]["terminal_class_path"] = 'paddle_rec_user_define.py'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.contains(self.out, 'User Define SingleInstance', self.err_msg)
        built_in.contains(self.out, 'User Define SingleNetwork', self.err_msg)
        built_in.contains(self.out, 'User Define SingleStartup', self.err_msg)
        built_in.contains(self.out, 'User Define SingleRunner', self.err_msg)
        built_in.contains(self.out, 'User Define SingleTerminal', self.err_msg)
