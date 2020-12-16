#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import six
sys.path.append(os.path.dirname(__file__))
import utils
import built_in
from paddle_rec_base import RankDNNBaseNewConfig


class TestRankDNNNewConfig(RankDNNBaseNewConfig):
    """test rank dnn model"""

    def __init__(self):
        super(TestRankDNNNewConfig, self).__init__()
        self.epoch_re = r'epoch.+done'
        self.run_time_re = r'.+use\stime:\s(.+)\n'
        self.batch_auc_re = r'BATCH_AUC:\s\[([0-9].[0-9]+)\]'
        self.auc_re = r'\bAUC:\s\[([0-9].[0-9]+)\]'

    def setUp(self):
        """do something for each cases."""
        utils.cmd_shell("rm -rf logs")
        # utils.cmd_shell("kill -9 `ps -ef|grep paddlerec|awk '{print $2}'`")

    """test single cpu cases"""
    def test_normal_c2(self):
        """test normal yaml construct by RankDNN base."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)

    def test_QueueDataset_train_c2(self):
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
                                   r'\ndataset.dataset_train.type\s+(\S+)\s+\n',
                                   check_type,
                                   self.err_msg)

    def test_workspace_abs_c2(self):
        """test abs worksapce."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["workspace"] = './PaddleRec/models/rank/dnn'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)

    def test_debug_open_c2(self):
        """test debug open."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["debug"] = 'True'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)

    def test_optimizer_adam_c2(self):
        """test optimizer adam."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["hyper_parameters"]['optimizer']['class'] = 'Adam'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)

    def test_increment_train_c2(self):
        """test increment train.
           both runners are single & train.
        """
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content['mode'] = ['runner0', 'runner1']
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 4, self.err_msg)

    def test_single_infer_in_epochs_dir_c2(self):
        """test single infer base on one epoch with new config.
        """
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content['mode'] = 'runner2'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.run_time_re, 1, self.err_msg)

#     def test_single_infer_in_base_dir_c2(self):
#         """test single infer base on save dir with new config.
#         """
#         self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
#         self.yaml_content['mode'] = 'runner2'
#         self.yaml_content['runner'][2]['init_model_path'] = 'increment_dnn'
#         self.run_yaml()
#         built_in.equals(self.pro.returncode, 0, self.err_msg)
#         built_in.not_contains(self.err, 'Traceback', self.err_msg)
#         built_in.regex_match_len(self.out, self.run_time_re, 2, self.err_msg)

    def test_two_phase_train_c2(self):
        """test single train with two phase in runner config.
        """
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["runner"][0]["phases"] = ['phase1', 'phase2']
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 4, self.err_msg)

    def test_infer_twice_c2(self):
        """test single infer twice and check the value.
        """
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content['mode'] = 'runner2'
        self.run_yaml()
        batch_auc1 = built_in.extract_value(self.out, self.batch_auc_re)
        auc1 = built_in.extract_value(self.out, self.auc_re)
        self.run_yaml()
        batch_auc2 = built_in.extract_value(self.out, self.batch_auc_re)
        auc2 = built_in.extract_value(self.out, self.auc_re)
        err_msg = "{} != {}".format(batch_auc1, batch_auc2)
        built_in.numpy_close(batch_auc1, batch_auc2, err_msg)
        err_msg = "{} != {}".format(auc1, auc2)
        built_in.numpy_close(auc1, auc2, err_msg)

    def test_mode_null_c2(self):
        """test mode is null c2."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content['mode'] = []
        self.run_yaml()
        built_in.equals(self.pro.returncode, 1, self.err_msg)
        built_in.contains(self.err, 'Traceback', self.err_msg)

    def test_runner_phases_empty_list_c2(self):
        """test phase is [] and it will run nothing."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["runner"][0]["phases"] = []
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_runner_no_phases_c2(self):
        """test runner has no phase and it will run all phase."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["runner"][0].pop("phases")
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 4, self.err_msg)
