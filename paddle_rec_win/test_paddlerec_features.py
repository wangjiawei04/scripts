#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(__file__))
from paddle_rec_base import RankDNNBase
import built_in
import sys
import utils
import six

class TestRankDNN(RankDNNBase):
    """test rank dnn model"""

    def __init__(self):
        super(TestRankDNN, self).__init__()
        self.epoch_re = r'epoch.+done'
        self.run_time_re = r'.+use\stime:\s(.+)\n'
        self.batch_auc_re = r'BATCH_AUC:\s\[([0-9].[0-9]+)\]'
        self.auc_re = r'\bAUC:\s\[([0-9].[0-9]+)\]'

    def setUp(self):
        """do something for each cases."""
        utils.cmd_shell("rm -rf logs")
        # utils.cmd_shell("rm -rf increment*")
        # utils.cmd_shell("rm -rf inference*")
        # utils.cmd_shell("kill -9 `ps -ef|grep paddlerec|awk '{print $2}'`")

    """test cpu cases"""
    def test_normal(self):
        """test normal yaml construct by RankDNN base."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)

    def test_QueueDataset_train(self):
        """test QueueDataset in train."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["dataset"][0]["type"] = "QueueDataset"
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)
        # NOTE windows和mac直接会强行切换到dataloader
        if utils.get_platform() != "LINUX" or not six.PY2:
            check_type = "DataLoader"
        else:
            check_type = "QueueDataset"
        built_in.regex_match_equal(self.out,
                                   '\ndataset.dataset_train.type\s+(\S+)\s+\n',
                                   check_type,
                                   self.err_msg)

    def test_workspace_abs(self):
        """test abs worksapce."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["workspace"] = 'models/rank/dnn'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)

    def test_debug_open(self):
        """test debug open."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["debug"] = 'True'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)

    def test_optimizer_adam(self):
        """test optimizer adam."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["hyper_parameters"]['optimizer']['class'] = 'Adam'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)

    def test_optimizer_sgd(self):
        """test optimizer sgd."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["hyper_parameters"]['optimizer']['class'] = 'SGD'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)

    def test_optimizer_sgd_reg(self):
        """test optimizer sgd with reg."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["hyper_parameters"]['optimizer']['class'] = 'SGD'
        self.yaml_content["hyper_parameters"]['reg'] = 0.1
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)

#     def test_optimizer_lr(self):
#         """test optimizer lr"""
#         self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
#         self.yaml_content["hyper_parameters"]['optimizer']['class'] = 'SGD'
#         self.yaml_content["hyper_parameters"]['optimizer']['learning_rate'] = 0.02
#         self.yaml_content["hyper_parameters"]['reg'] = 0.1
#         self.run_yaml()
#         built_in.equals(self.pro.returncode, 0, self.err_msg)
#         built_in.not_contains(self.err, 'Traceback', self.err_msg)
#         built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)
#         built_in.regex_match_equal(self.out,
#                                    '\nhyper_parameters.optimizer.learning_rate\s+(\S+)\s+\n',
#                                    '0.02',
#                                    self.err_msg)

    def test_increment_train(self):
        """test increment train."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content['mode'] = 'runner1'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)
        built_in.regex_match_equal(self.out, '\nmode\s+(\S+)\s+\n', 'runner1', self.err_msg)

    def test_single_infer(self):
        """test single infer."""
        # run basic to save models
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.run_yaml()
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content['mode'] = 'runner2'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.run_time_re, 1, self.err_msg)

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
        built_in.regex_match_len(self.out, self.run_time_re, 4, self.err_msg)

#     def test_thread_num(self):
#         """test thread num."""
#         self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
#         self.yaml_content['phase'].append({
#             'name': 'phase2',
#             'model': '{workspace}/model.py',  # user-defined model
#             'dataset_name': 'dataset_infer',  # select dataset by name
#             'thread_num': 2
#         })
#         self.run_yaml()
#         built_in.equals(self.pro.returncode, 0, self.err_msg)
#         built_in.not_contains(self.err, 'Traceback', self.err_msg)
#         built_in.regex_match_len(self.out, self.run_time_re, 4, self.err_msg)
#         built_in.regex_match_equal(self.out,
#                                    '\nphase.phase2.thread_num\s+(\S+)\s+\n',
#                                    2,
#                                    self.err_msg)

    def test_infer_twice(self):
        """test infer twice."""
        # run basic to save models
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.run_yaml()
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
        l1 = built_in.extract_value(self.out, self.auc_re)
        self.run_yaml()
        l2 = built_in.extract_value(self.out, self.auc_re)
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
        total_list = built_in.extract_value(self.out, self.auc_re)
        err_msg = "{} != {}".format(total_list[0:1], total_list[2:3])
        built_in.numpy_close(total_list[0:1], total_list[2:3], err_msg)


    """test gpu cases."""
    def test_normal_gpu(self):
        """test normal yaml construct by RankDNN base in gpu."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["runner"][0]["device"] = 'gpu'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)

    def test_QueueDataset_train_gpu(self):
        """test QueueDataset in train  with gpu"""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["runner"][0]["device"] = 'gpu'
        self.yaml_content["dataset"][0]["type"] = "QueueDataset"
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)
        # NOTE windows和mac直接会强行切换到dataloader
        if utils.get_platform() != "LINUX" or not six.PY2:
            check_type = "DataLoader"
        else:
            check_type = "QueueDataset"
        built_in.regex_match_equal(self.out,
                                   '\ndataset.dataset_train.type\s+(\S+)\s+\n',
                                   check_type,
                                   self.err_msg)

    def test_debug_open_gpu(self):
        """test debug open with gpu."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["runner"][0]["device"] = 'gpu'
        self.yaml_content["debug"] = 'True'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)

    def test_optimizer_adam_gpu(self):
        """test optimizer adam with gpu."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["runner"][0]["device"] = 'gpu'
        self.yaml_content["hyper_parameters"]['optimizer']['class'] = 'Adam'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)

    def test_optimizer_sgd_gpu(self):
        """test optimizer sgd with gpu."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["runner"][0]["device"] = 'gpu'
        self.yaml_content["hyper_parameters"]['optimizer']['class'] = 'SGD'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)

    def test_optimizer_sgd_reg_gpu(self):
        """test optimizer sgd and reg with gpu. """
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["runner"][0]["device"] = 'gpu'
        self.yaml_content["hyper_parameters"]['optimizer']['class'] = 'SGD'
        self.yaml_content["hyper_parameters"]['reg'] = 0.1
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)

#     def test_optimizer_lr_gpu(self):
#         """test optimizer lr with gpu."""
#         self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
#         self.yaml_content["runner"][0]["device"] = 'gpu'
#         self.yaml_content["hyper_parameters"]['optimizer']['class'] = 'SGD'
#         self.yaml_content["hyper_parameters"]['optimizer']['learning_rate'] = 0.02
#         self.yaml_content["hyper_parameters"]['reg'] = 0.1
#         self.run_yaml()
#         built_in.equals(self.pro.returncode, 0, self.err_msg)
#         built_in.not_contains(self.err, 'Traceback', self.err_msg)
#         built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)
#         built_in.regex_match_equal(self.out,
#                                    '\nhyper_parameters.optimizer.learning_rate\s+(\S+)\s+\n',
#                                    '0.02',
#                                    self.err_msg)

    def test_increment_train_gpu(self):
        """test increment train with gpu."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content['mode'] = 'runner1'
        self.yaml_content["runner"][1]["device"] = 'gpu'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)
        built_in.regex_match_equal(self.out, '\nmode\s+(\S+)\s+\n', 'runner1', self.err_msg)

    def test_single_infer_gpu(self):
        """test single infer with gpu."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content['mode'] = 'runner2'
        self.yaml_content["runner"][2]["device"] = 'gpu'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.run_time_re, 1, self.err_msg)

    def test_two_phase_train_gpu(self):
        """test two phase train with gpu."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["runner"][0]["device"] = 'gpu'
        self.yaml_content['phase'].append({
            'name': 'phase2',
            'model': '{workspace}/model.py',  # user-defined model
            'dataset_name': 'dataset_infer',  # select dataset by name
            'thread_num': 1
        })
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.run_time_re, 4, self.err_msg)

#     def test_thread_num_gpu(self):
#         """test thread num with gpu."""
#         self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
#         self.yaml_content["runner"][0]["device"] = 'gpu'
#         self.yaml_content['phase'].append({
#             'name': 'phase2',
#             'model': '{workspace}/model.py',  # user-defined model
#             'dataset_name': 'dataset_infer',  # select dataset by name
#             'thread_num': 2
#         })
#         self.run_yaml()
#         built_in.equals(self.pro.returncode, 0, self.err_msg)
#         built_in.not_contains(self.err, 'Traceback', self.err_msg)
#         built_in.regex_match_len(self.out, self.epoch_re, 4, self.err_msg)
#         built_in.regex_match_equal(self.out,
#                                    '\nphase.phase2.thread_num\s+(\S+)\s+\n',
#                                    2,
#                                    self.err_msg)

    def test_infer_twice_gpu(self):
        """test infer twice with gpu."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content['phase'].pop()
        self.yaml_content["mode"] = "runner2"
        self.yaml_content["runner"][2]["device"] = 'gpu'
        self.yaml_content['phase'].append({
            'name': 'phase2',
            'model': '{workspace}/model.py',  # user-defined model
            'dataset_name': 'dataset_infer',  # select dataset by name
            'thread_num': 2
        })
        self.run_yaml()
        l1 = built_in.extract_value(self.out, self.auc_re)
        self.run_yaml()
        l2 = built_in.extract_value(self.out, self.auc_re)
        err_msg = "{} != {}".format(l1, l2)
        built_in.numpy_close(l1, l2, err_msg)

    def test_two_phase_infer_gpu(self):
        """test two infer in phase with gpu."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content['phase'].pop()
        self.yaml_content["mode"] = "runner2"
        self.yaml_content["runner"][2]["device"] = 'gpu'
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
        total_list = built_in.extract_value(self.out, self.auc_re)
        err_msg = "{} != {}".format(total_list[0:1], total_list[2:3])
        built_in.numpy_close(total_list[0:1], total_list[2:3], err_msg)
