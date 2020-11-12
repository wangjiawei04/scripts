#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import six
import sys
#import utils
sys.path.append(os.path.dirname(__file__))
import utils
import built_in
from paddle_rec_mmoe import MultiTaskMMOEBase


class TestMMOE(MultiTaskMMOEBase):
    """test MultiTask MMOE model"""

    """test cpu cases"""
    def test_mmoe_normal(self):
        """test normal yaml construct by MultiTaskMMOE base."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, 'epoch.+done', 2, self.err_msg)

    def test_QueueDataset_train(self):
        """test QueueDataset in train."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        if six.PY3:
           self.yaml_content["dataset"][0]["type"] = "DataLoader"
        else:
           self.yaml_content["dataset"][0]["type"] = "QueueDataset"
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, 'epoch.+done', 2, self.err_msg)
        # NOTE windows和mac直接会强行切换到dataloader
        check_type = "DataLoader" if utils.get_platform() != "LINUX" else "QueueDataset"
        if six.PY3:
           check_type = "DataLoader"
        else:
           check_type = "DataLoader" if utils.get_platform() != "LINUX" else "QueueDataset"
   #     built_in.regex_match_equal(self.out,
   #                                '\ndataset.dataset_train.type\s+(\S+)\s+\n',
   #                                check_type,
   #                                self.err_msg)

    def test_debug_open(self):
        """test debug open."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["debug"] = 'True'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, 'epoch.+done', 2, self.err_msg)

    def test_optimizer_adam(self):
        """test optimizer adam."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["hyper_parameters"]['optimizer']['class'] = 'Adam'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, 'epoch.+done', 2, self.err_msg)

    def test_optimizer_sgd(self):
        """test optimizer sgd."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["hyper_parameters"]['optimizer']['class'] = 'SGD'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, 'epoch.+done', 2, self.err_msg)

    def test_optimizer_sgd_reg(self):
        """test optimizer sgd with reg."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["hyper_parameters"]['optimizer']['class'] = 'SGD'
        self.yaml_content["hyper_parameters"]['reg'] = 0.1
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, 'epoch.+done', 2, self.err_msg)

    def test_optimizer_lr(self):
        """test optimizer lr"""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["hyper_parameters"]['optimizer']['class'] = 'SGD'
        self.yaml_content["hyper_parameters"]['optimizer']['learning_rate'] = 0.02
        self.yaml_content["hyper_parameters"]['reg'] = 0.1
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, 'epoch.+done', 2, self.err_msg)
        built_in.regex_match_equal(self.out,
                                   '\nhyper_parameters.optimizer.learning_rate\s+(\S+)\s+\n',
                                   '0.02',
                                   self.err_msg)

    def test_increment_train(self):
        """test increment train."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content['mode'] = 'runner1'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, 'epoch.+done', 2, self.err_msg)
        built_in.regex_match_equal(self.out, '\nmode\s+(\S+)\s+\n', 'runner1', self.err_msg)

    def test_single_infer(self):
         """test single infer."""
         self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
         self.yaml_content['mode'] = 'runner2'

         self.run_yaml()
         built_in.equals(self.pro.returncode, 0, self.err_msg)
         built_in.not_contains(self.err, 'Traceback', self.err_msg)
        # built_in.regex_match_len(self.out, 'Infer.+done', 1, self.err_msg)

    def test_two_phase_train(self):
        """test two phase train"""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content['phase'].append({
            'name': 'phase2',
            'model': '{workspace}/model.py',  # user-defined model
            'dataset_name': 'dataset_infer',  # select dataset by name
            'thread_num': 1
        })
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, 'epoch.+done', 4, self.err_msg)

    def test_thread_num(self):
        """test thread num."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content['phase'].append({
            'name': 'phase2',
            'model': '{workspace}/model.py',  # user-defined model
            'dataset_name': 'dataset_infer',  # select dataset by name
            'thread_num': 2
        })
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, 'epoch.+done', 4, self.err_msg)
        built_in.regex_match_equal(self.out,
                                   '\nphase.phase2.thread_num\s+(\S+)\s+\n',
                                   2,
                                   self.err_msg)

    def test_infer_twice(self):
        """test infer twice."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content['phase'].pop()
        self.yaml_content["mode"] = "runner2"
        self.yaml_content['phase'].append({
            'name': 'phase2',
            'model': '{workspace}/model.py',  # user-defined model
            'dataset_name': 'dataset_infer',  # select dataset by name
            'thread_num': 2
        })
        self.run_yaml()
        l1 = built_in.extract_value(self.out, r'.+,\sAUC_marital:\s\[(.+)\],')
        self.run_yaml()
        l2 = built_in.extract_value(self.out, r'.+,\sAUC_marital:\s\[(.+)\],')
        err_msg = "{} != {}".format(l1, l2)
        built_in.numpy_close(l1, l2, err_msg)

    def test_two_phase_infer(self):
        """test two infer in phase."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content['phase'].pop()
        self.yaml_content["mode"] = "runner2"
        self.yaml_content['phase'].append({
            'name': 'phase2',
            'model': '{workspace}/model.py',  # user-defined model
            'dataset_name': 'dataset_infer',  # select dataset by name
            'thread_num': 2
        })
        self.yaml_content['phase'].append({
            'name': 'phase2',
            'model': '{workspace}/model.py',  # user-defined model
            'dataset_name': 'dataset_infer',  # select dataset by name
            'thread_num': 2
        })
        self.run_yaml()
        total_list = built_in.extract_value(self.out, r'.+,\sAUC_marital:\s\[(.+)\],')
        err_msg = "{} != {}".format(total_list[::2], total_list[1::2])
        built_in.numpy_close(total_list[::2], total_list[1::2], err_msg)
