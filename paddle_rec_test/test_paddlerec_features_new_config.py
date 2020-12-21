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
        self.auc_re_bug = r'\bAUC:\s\[([0-9].[0-9]+)\]'

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
    #    built_in.regex_match_equal(self.out,
    #                               r'\ndataset.dataset_train.type\s+(\S+)\s+\n',
    #                               check_type,
   #                                self.err_msg)

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

    # geo 不再支持adam，只能用sgd
    # def test_optimizer_adam_c2(self):
    #     """test optimizer adam."""
    #     self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
    #     self.yaml_content["hyper_parameters"]['optimizer']['class'] = 'Adam'
    #     self.run_yaml()
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains(self.err, 'Traceback', self.err_msg)
    #     built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)

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
        # built_in.regex_match_len(self.out, self.run_time_re, 1, self.err_msg)

    def test_single_infer_in_base_dir_c2(self):
        """test single infer base on save dir with new config.
        """
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content['mode'] = 'runner2'
        self.yaml_content['runner'][2]['init_model_path'] = 'increment_dnn'
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        # built_in.regex_match_len(self.out, self.run_time_re, 2, self.err_msg)

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

    # def test_mode_null_c2(self):
    #     """test mode is null c2."""
    #     self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
    #     self.yaml_content['mode'] = []
    #     self.run_yaml()
    #     built_in.equals(self.pro.returncode, 1, self.err_msg)
    #     built_in.contains(self.err, 'Traceback', self.err_msg)

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

    """test ps cases."""
    # def test_mode_list_ps_local_cluster_c2(self):
    #     """test mode list has one element and the runner is local cluster.
    #     """
    #     utils.cmd_shell("rm -rf logs")
    #     self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
    #     self.yaml_content["runner"][0]["class"] = "local_cluster_train"
    #     self.run_yaml()
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains(self.err, 'Traceback', self.err_msg)
    #
    # def test_mode_list_ps_local_cluster_and_increment_c2(self):
    #     """test mode list has two elements and both runner are local cluster.
    #        one is train, the other is increment training.
    #     """
    #     self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
    #     self.yaml_content["runner"][0]["class"] = "local_cluster_train"
    #     self.yaml_content["runner"][1]["class"] = "local_cluster_train"
    #     self.yaml_content["mode"] = ["runner0", "runner1"]
    #     self.run_yaml()
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains(self.err, 'Traceback', self.err_msg)
    #     built_in.regex_match_len('logs/worker.0', self.epoch_re, 2, self.err_msg)

#     def test_mode_list_ps_local_cluster_and_infer_c2(self):
#         """test mode list has two elements .
#            one is local cluster and one is infer.
#         """
#         self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
#         self.yaml_content["runner"][0]["class"] = "local_cluster_train"
#         self.yaml_content["mode"] = ["runner0", "runner2"]
#         self.run_yaml()
#         built_in.equals(self.pro.returncode, 0, self.err_msg)
#         built_in.not_contains(self.err, 'Traceback', self.err_msg)
#         built_in.regex_match_len(self.out, self.auc_re_bug, 0, self.err_msg)

    # def test_mode_str_ps_local_cluster_1p_1t_async_c2(self):
    #     """test_mode_str_ps_local_cluster_1p_1t_c2."""
    #     self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
    #     self.yaml_content["mode"] = "runner0"
    #     self.yaml_content["runner"][0]["class"] = "local_cluster_train"
    #     self.run_yaml()
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains('logs/server.0', 'Traceback', self.err_msg)
    #     built_in.contains('logs/worker.0', 'AsyncCommunicator Initialized', self.err_msg)
    #
    # def test_mode_str_ps_local_cluster_1p_1t_sync_c2(self):
    #     """test_mode_str_ps_local_cluster_1p_1t_sync_c2."""
    #     self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
    #     self.yaml_content["mode"] = "runner0"
    #     self.yaml_content["runner"][0]["class"] = "local_cluster_train"
    #     self.yaml_content["runner"][0]["distribute_strategy"] = "sync"
    #     self.run_yaml()
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains('logs/server.0', 'Traceback', self.err_msg)
    #     built_in.contains('logs/worker.0', 'SyncCommunicator Initialized', self.err_msg)

    # def test_mode_str_ps_local_cluster_1p_1t_half_async_c2(self):
    #     """test_mode_str_ps_local_cluster_1p_1t_half_async_c2."""
    #     self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
    #     self.yaml_content["mode"] = "runner0"
    #     self.yaml_content["runner"][0]["class"] = "local_cluster_train"
    #     self.yaml_content["runner"][0]["distribute_strategy"] = "half_async"
    #     self.run_yaml()
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.regex_match_len('logs/worker.0', self.epoch_re, 2, self.err_msg)
    #     built_in.contains('logs/worker.0', 'HalfAsyncCommunicator Initialized', self.err_msg)

    # def test_mode_str_ps_local_cluster_1p_1t_geo_c2(self):
    #     """test_mode_str_ps_local_cluster_1p_1t_geo_c2."""
    #     self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
    #     self.yaml_content["mode"] = "runner0"
    #     self.yaml_content["runner"][0]["class"] = "local_cluster_train"
    #     self.yaml_content["runner"][0]["distribute_strategy"] = "geo"
    #     self.run_yaml()
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.regex_match_len('logs/worker.0', self.epoch_re, 2, self.err_msg)
    #     built_in.contains('logs/worker.0', 'GeoSgdCommunicator Initialized', self.err_msg)

    # def test_mode_str_ps_local_cluster_1p_2t_1f_async_c2(self):
    #     """test_mode_str_ps_local_cluster_1p_2t_c2."""
    #     self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
    #     self.yaml_content["mode"] = "runner0"
    #     self.yaml_content["runner"][0]["class"] = "local_cluster_train"
    #     self.yaml_content["runner"][0]["worker_num"] = 2
    #     self.yaml_content["runner"][0]["server_num"] = 1
    #     self.run_yaml()
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains('logs/server.0', 'Traceback', self.err_msg)
    #     built_in.not_contains('logs/worker.0', 'Traceback', self.err_msg)
    #     built_in.path_not_exist('logs/worker.1', self.err_msg)
    #
    # def test_mode_str_ps_local_cluster_1p_2t_2f_async_c2(self):
    #     """test_mode_str_ps_local_cluster_1p_2t_c2.
    #        worker_num会被data_path下得文件数量来覆盖掉；
    #     """
    #     self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
    #     self.yaml_content["mode"] = "runner0"
    #     self.yaml_content["runner"][0]["class"] = "local_cluster_train"
    #     self.yaml_content["runner"][0]["worker_num"] = 10
    #     self.yaml_content["runner"][0]["server_num"] = 1
    #     self.yaml_content["dataset"][0]["data_path"] = "criteo_data"
    #     self.run_yaml()
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains('logs/server.0', 'Traceback', self.err_msg)
    #     built_in.not_contains('logs/worker.0', 'Traceback', self.err_msg)
    #     built_in.regex_match_len('logs/worker.1', self.epoch_re, 2, self.err_msg)
    #
    # def test_mode_str_ps_local_cluster_1p_2t_sync_c2(self):
    #     """test_mode_str_ps_local_cluster_1p_2t_sync_c2."""
    #     self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
    #     self.yaml_content["mode"] = "runner0"
    #     self.yaml_content["runner"][0]["class"] = "local_cluster_train"
    #     self.yaml_content["runner"][0]["distribute_strategy"] = "sync"
    #     self.yaml_content["runner"][0]["worker_num"] = 2
    #     self.yaml_content["runner"][0]["server_num"] = 1
    #     self.run_yaml()
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains('logs/server.0', 'Traceback', self.err_msg)

    # half async在2.0被废弃掉
    # def test_mode_str_ps_local_cluster_1p_2t_half_async_c2(self):
    #     """test_mode_str_ps_local_cluster_1p_2t_half_async_c2."""
    #     self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
    #     self.yaml_content["mode"] = "runner0"
    #     self.yaml_content["runner"][0]["class"] = "local_cluster_train"
    #     self.yaml_content["runner"][0]["distribute_strategy"] = "half_async"
    #     self.yaml_content["runner"][0]["worker_num"] = 2
    #     self.yaml_content["runner"][0]["server_num"] = 1
    #     self.yaml_content["dataset"][0]["data_path"] = "criteo_data"
    #     self.run_yaml()
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.regex_match_len('logs/worker.0', self.epoch_re, 2, self.err_msg)
    #     built_in.regex_match_len('logs/worker.1', self.epoch_re, 2, self.err_msg)

    # def test_mode_str_ps_local_cluster_1p_2t_geo_c2(self):
    #     """test_mode_str_ps_local_cluster_1p_2t_geo_c2."""
    #     self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
    #     self.yaml_content["mode"] = "runner0"
    #     self.yaml_content["runner"][0]["class"] = "local_cluster_train"
    #     self.yaml_content["runner"][0]["distribute_strategy"] = "geo"
    #     self.yaml_content["runner"][0]["worker_num"] = 2
    #     self.yaml_content["runner"][0]["server_num"] = 1
    #     self.yaml_content["dataset"][0]["data_path"] = "criteo_data"
    #     self.run_yaml()
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.regex_match_len('logs/worker.0', self.epoch_re, 2, self.err_msg)
    #     built_in.regex_match_len('logs/worker.1', self.epoch_re, 2, self.err_msg)

    # def test_mode_str_ps_local_cluster_2p_2t_async_c2(self):
    #     """test_mode_str_ps_local_cluster_2p_2t_async_c2."""
    #     self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
    #     self.yaml_content["mode"] = "runner0"
    #     self.yaml_content["runner"][0]["class"] = "local_cluster_train"
    #     self.yaml_content["runner"][0]["worker_num"] = 2
    #     self.yaml_content["runner"][0]["server_num"] = 2
    #     self.yaml_content["dataset"][0]["data_path"] = "criteo_data"
    #     self.run_yaml()
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains('logs/server.0', 'Traceback', self.err_msg)
    #     built_in.not_contains('logs/server.1', 'Traceback', self.err_msg)
    #     built_in.regex_match_len('logs/worker.0', self.epoch_re, 2, self.err_msg)
    #     built_in.regex_match_len('logs/worker.1', self.epoch_re, 2, self.err_msg)
    #
    # def test_mode_str_ps_local_cluster_2p_2t_sync_c2(self):
    #     """test_mode_str_ps_local_cluster_2p_2t_sync_c2."""
    #     self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
    #     self.yaml_content["mode"] = "runner0"
    #     self.yaml_content["runner"][0]["class"] = "local_cluster_train"
    #     self.yaml_content["runner"][0]["distribute_strategy"] = "sync"
    #     self.yaml_content["runner"][0]["worker_num"] = 2
    #     self.yaml_content["runner"][0]["server_num"] = 2
    #     self.yaml_content["dataset"][0]["data_path"] = "criteo_data"
    #     self.run_yaml()
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains('logs/server.0', 'Traceback', self.err_msg)
    #     built_in.not_contains('logs/server.1', 'Traceback', self.err_msg)
    #     built_in.regex_match_len('logs/worker.0', self.epoch_re, 2, self.err_msg)
    #     built_in.regex_match_len('logs/worker.1', self.epoch_re, 2, self.err_msg)

    # def test_mode_str_ps_local_cluster_2p_2t_half_async_c2(self):
    #     """test_mode_str_ps_local_cluster_2p_2t_half_async_c2."""
    #     self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
    #     self.yaml_content["mode"] = "runner0"
    #     self.yaml_content["runner"][0]["class"] = "local_cluster_train"
    #     self.yaml_content["runner"][0]["distribute_strategy"] = "half_async"
    #     self.yaml_content["runner"][0]["worker_num"] = 2
    #     self.yaml_content["runner"][0]["server_num"] = 2
    #     self.yaml_content["dataset"][0]["data_path"] = "criteo_data"
    #     self.run_yaml()
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains('logs/server.0', 'Traceback', self.err_msg)
    #     built_in.not_contains('logs/server.1', 'Traceback', self.err_msg)
    #     built_in.regex_match_len('logs/worker.0', self.epoch_re, 2, self.err_msg)
    #     built_in.regex_match_len('logs/worker.1', self.epoch_re, 2, self.err_msg)

    # def test_mode_str_ps_local_cluster_2p_2t_geo_c2(self):
    #     """test_mode_str_ps_local_cluster_2p_2t_geo_c2."""
    #     self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
    #     self.yaml_content["mode"] = "runner0"
    #     self.yaml_content["runner"][0]["class"] = "local_cluster_train"
    #     self.yaml_content["runner"][0]["distribute_strategy"] = "geo"
    #     self.yaml_content["runner"][0]["worker_num"] = 2
    #     self.yaml_content["runner"][0]["server_num"] = 2
    #     self.yaml_content["dataset"][0]["data_path"] = "criteo_data"
    #     self.run_yaml()
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains('logs/server.0', 'Traceback', self.err_msg)
    #     built_in.not_contains('logs/server.1', 'Traceback', self.err_msg)
    #     built_in.regex_match_len('logs/worker.0', self.epoch_re, 2, self.err_msg)
    #     built_in.regex_match_len('logs/worker.1', self.epoch_re, 2, self.err_msg)

    """test collective."""
    def test_mode_list_single_selected_gpus_1card_c2(self):
        """test selected gpus 1card, it will run with single mode."""
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["runner"][0]["device"] = 'gpu'
        self.yaml_content["runner"][0]["selected_gpus"] = "0"
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len(self.out, self.epoch_re, 2, self.err_msg)
        built_in.regex_match_equal(self.out,
                                   '\ntrain.trainer.engine\s+(\S+)\s+\n',
                                   "single",
                                   self.err_msg)
    
    def test_mode_list_ps_selected_gpus_2f_2card_c2(self):
        """test selected gpus 2card with two files and not set fleet mode,
           it will change ps to collective and run with local_cluster_train mode
        """
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["runner"][0]["device"] = 'gpu'
        self.yaml_content["runner"][0]["selected_gpus"] = "0,1"
        self.yaml_content["dataset"][0]["data_path"] = "criteo_data"
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.path_not_exist('logs/server.0', self.err_msg)
        built_in.regex_match_equal(self.out,
                                   '\ntrain.trainer.engine\s+(\S+)\s+\n',
                                   "local_cluster",
                                   self.err_msg)
        if six.PY2:
            built_in.regex_match_len('logs/worker.1', self.auc_re, 7, self.err_msg)
#         elif six.PY3:
#             built_in.regex_match_len('logs/worker.1', self.auc_re, 8, self.err_msg)

    # NOTE: this case is error, open if after fixed
    # def test_mode_list_collective_selected_gpus_1f_2cards_c2(self):
    #     """
    #     test_collective_selected_gpus_1f_2cards.
    #     程序运行GPU卡号，若以"0,1"的方式指定多卡，则会默认启用collective模式
    #     测试只有1份数据得时候，即使selected_gpus为2，也只有1个trainer.
    #     """
    #     self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
    #     self.yaml_content["runner"][0]["device"] = 'gpu'
    #     self.yaml_content["runner"][0]["selected_gpus"] = "0,1"
    #     self.yaml_content["runner"][0]["class"] = "local_cluster_train"
    #     self.yaml_content["runner"][0]["fleet_mode"] = "collective"
    #     self.run_yaml()
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains(self.err, 'Traceback', self.err_msg)
    #     built_in.regex_match_len('logs/worker.0', self.epoch_re, 2, self.err_msg)
    #     built_in.path_not_exist('logs/worker.1', self.err_msg)


    def test_mode_list_collective_selected_gpus_2f_2cards_c2(self):
        """test selected gpus 2card with two files and set fleet mode = collective,
           it will change ps to collective and run with local_cluster_train mode
        """
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["runner"][0]["device"] = 'gpu'
        self.yaml_content["runner"][0]["selected_gpus"] = "0,1"
        self.yaml_content["runner"][0]["class"] = "local_cluster_train"
        self.yaml_content["runner"][0]["fleet_mode"] = "collective"
        self.yaml_content["dataset"][0]["data_path"] = "criteo_data"
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len('logs/worker.0', self.epoch_re, 2, self.err_msg)
        built_in.path_not_exist('logs/server.0', self.err_msg)

    def test_mode_list_collective_selected_gpus_2f_4cards_c2(self):
        """
        test_collective_selected_gpus_2f_2cards.
        程序运行GPU卡号，会依据文件个数, gpu_nums, worker_num 来判断起多少个trainer.
        """
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["runner"][0]["device"] = 'gpu'
        self.yaml_content["runner"][0]["selected_gpus"] = "0,1"
        self.yaml_content["runner"][0]["class"] = "local_cluster_train"
        self.yaml_content["runner"][0]["fleet_mode"] = "collective"
        self.yaml_content["dataset"][0]["data_path"] = "criteo_data"
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)
        built_in.regex_match_len('logs/worker.1', self.epoch_re, 2, self.err_msg)
        built_in.path_not_exist('logs/worker.2', self.err_msg)

    def test_mode_list_ps_selected_gpus_2cards_c2(self):
        """
        test fleet_mode is ps and device is gpu, it will raise error.；
        """
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["runner"][0]["device"] = 'gpu'
        self.yaml_content["runner"][0]["selected_gpus"] = "0,1"
        self.yaml_content["runner"][0]["class"] = "local_cluster_train"
        self.yaml_content["runner"][0]["fleet_mode"] = "ps"
        self.run_yaml()
        built_in.not_equals(self.pro.returncode, 1, self.err_msg)

    def test_mode_list_collective_local_cluster_and_increment_c2(self):
        """test mode list has two elements and both runner are local cluster.
           one is train, the other is increment training.
        """
        self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
        self.yaml_content["runner"][0]["device"] = 'gpu'
        self.yaml_content["runner"][0]["selected_gpus"] = "0,1"
        self.yaml_content["runner"][0]["class"] = "local_cluster_train"
        self.yaml_content["runner"][0]["fleet_mode"] = "collective"
        self.yaml_content["dataset"][0]["data_path"] = "criteo_data"
        self.yaml_content["runner"][1]["device"] = 'gpu'
        self.yaml_content["runner"][1]["selected_gpus"] = "0,1"
        self.yaml_content["runner"][1]["class"] = "local_cluster_train"
        self.yaml_content["runner"][1]["fleet_mode"] = "collective"
        self.yaml_content["dataset"][1]["data_path"] = "criteo_data"
        self.yaml_content["mode"] = ["runner0", "runner1"]
        self.run_yaml()
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        # built_in.not_contains(self.err, 'Traceback', self.err_msg)
        # built_in.regex_match_len('logs/worker.0', self.epoch_re, 2, self.err_msg)
        # built_in.regex_match_len('logs/worker.1', self.epoch_re, 2, self.err_msg)
        # built_in.regex_match_len('logs/worker.1', '.+load.+increment_dnn', 1, self.err_msg)

#     def test_mode_list_collective_local_cluster_and_infer_c2(self):
#         """test mode list has two elements .
#            one is local cluster and one is infer.
#         """
#         self.yaml_config_name = sys._getframe().f_code.co_name + '.yaml'
#         self.yaml_content["runner"][0]["device"] = 'gpu'
#         self.yaml_content["runner"][0]["selected_gpus"] = "0,1"
#         self.yaml_content["runner"][0]["class"] = "local_cluster_train"
#         self.yaml_content["runner"][0]["fleet_mode"] = "collective"
#         self.yaml_content["dataset"][0]["data_path"] = "criteo_data"
#         self.yaml_content["runner"][2]["device"] = 'gpu'
#         self.yaml_content["mode"] = ["runner0", "runner2"]
#         self.run_yaml()
#         built_in.equals(self.pro.returncode, 0, self.err_msg)
#         built_in.not_contains(self.err, 'Traceback', self.err_msg)
#         built_in.regex_match_len(self.out, self.auc_re_bug, 0, self.err_msg)
