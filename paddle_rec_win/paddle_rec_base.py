#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import six
import abc
import subprocess
import nose.tools as tools
from utils import generate_yaml_file

class PaddleRecBase(object):
    """paddle rec base."""
    def __init__(self):
        self.yaml_content = {}
        self._construct_global_vars()
        self._construct_runners()
        self._construct_phase()
        self._construct_dataset()
        self._construct_hyper_paramters()
        self.yaml_config_name = './test.yaml'

    @abc.abstractmethod
    def _construct_global_vars(self):
        """abs _construct_global_vars"""
        pass

    @abc.abstractmethod
    def _construct_runners(self):
        """abs _construct_runners"""
        pass

    @abc.abstractmethod
    def _construct_phase(self):
        """abs _construct_phase"""
        pass

    @abc.abstractmethod
    def _construct_dataset(self):
        """abs _construct_dataset"""
        pass

    @abc.abstractmethod
    def _construct_hyper_paramters(self):
        """abs _construct_hyper_paramters"""
        pass

    def run_yaml(self, generate=True, cuda_devices=0):
        """
        run paddlerc yaml file to get result
        Returns:
        """
        cmd = ""
        if generate:
            generate_yaml_file(dict(self.yaml_content), self.yaml_config_name)
        if cuda_devices:
#            cmd = "CUDA_VISIBLE_DEVICES={} ".format(cuda_devices)
            cmd = ''
        cmd += 'python -m paddlerec.run -m {}'.format(self.yaml_config_name)
        print("run cmd is: {}".format(cmd))
        self.pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.out, self.err = self.pro.communicate()
        self.err_msg = "job failed, yaml is: ******************************************\n{}\n" \
                       "********** error is:  **************************\n{}".format(self.yaml_content,
                                                                      self.err)


class RankDNNBase(PaddleRecBase):
    """RankDNN test"""
    def __init__(self):
        super(RankDNNBase, self).__init__()

    def _construct_global_vars(self):
        self.yaml_content['debug'] = True
        self.yaml_content['workspace'] = "models/rank/dnn"
        self.yaml_content['mode'] = 'runner0'


    def _construct_runners(self):
        """construct_runners of RankDNN"""
        self.yaml_content['runner'] = []
        self.yaml_content['runner'].append({
            'name': 'runner0',
            'class': 'train',
            'device': 'cpu',
            'epochs': 2,
            'init_model_path': '',
            'save_checkpoint_interval': 2,
            'save_checkpoint_path': 'increment_dnn',
            'save_inference_interval': 2,
            'save_inference_path': 'inference_dnn',
            'save_inference_feed_varnames': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                             '11', '12', '13', '14', '15', '16', '17', '18', '19',
                                             '20', '21', '22', '23', '24', '25', '26', 'dense_var'],
            'save_inference_fetch_varnames': ['fc_3.tmp_1'],
            'print_interval': 10,
        })
        self.yaml_content['runner'].append({
            'name': 'runner1',
            'class': 'train',
            'device': 'cpu',
            'epochs': 2,
            'init_model_path': 'increment_dnn/1'
        })

        self.yaml_content['runner'].append({
            'name': 'runner2',
            'class': 'infer',
            'device': 'cpu',
            'epochs': 2,
            'init_model_path': 'increment_dnn/1'
        })

    def _construct_phase(self):
        """_construct_phase of RankDNN"""
        self.yaml_content['phase'] = []
        self.yaml_content['phase'].append({
            'name': 'phase1',
            'model': '{workspace}/model.py',  # user-defined model
            'dataset_name': 'dataset_train',  # select dataset by name
            'thread_num': 1
        })

    def _construct_dataset(self):
        """_construct_dataset of RankDNN"""
        self.yaml_content['dataset'] = []
        self.yaml_content['dataset'].append({
            'name': 'dataset_train',
            'dense_slots': 'dense_var:13',
            'sparse_slots': 'click 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26',
            'data_path': '{workspace}/data/sample_data/train',
            'type': 'DataLoader',  # 指定数据读取方式(DataLoader / QueueDataset)
            'batch_size': 2
            })
        self.yaml_content['dataset'].append({
            'dense_slots': 'dense_var:13',
            'name': 'dataset_infer',
            'sparse_slots': 'click 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26',
            'data_path': '{workspace}/data/sample_data/train',
            'type': 'DataLoader',
            'batch_size': 2})

    def _construct_hyper_paramters(self):
        """_construct_hyper_paramters of RankDNN"""
        self.yaml_content['hyper_parameters'] = {
            'sparse_feature_number': 1000001,
            'sparse_feature_dim': 9,
            'fc_sizes': [512, 256, 128, 64, 32],
            'optimizer': {
                'class': 'Adagrad',
                'learning_rate': 0.0001,
                'strategy': 'sync'
            }
        }

class RankDNNBaseNewConfig(PaddleRecBase):
    """RankDNNBaseNewConfig test"""
    def __init__(self):
        super(RankDNNBaseNewConfig, self).__init__()

    def _construct_global_vars(self):
        self.yaml_content['debug'] = True
        self.yaml_content['workspace'] = "models/rank/dnn"
        self.yaml_content['mode'] = ['runner0']


    def _construct_runners(self):
        """construct_runners of RankDNN"""
        self.yaml_content['runner'] = []
        self.yaml_content['runner'].append({
            'name': 'runner0',
            'class': 'train',
            'device': 'cpu',
            'epochs': 2,
            'init_model_path': '',
            'save_checkpoint_interval': 1,
            'save_checkpoint_path': 'increment_dnn',
            'save_inference_interval': 2,
            'save_inference_path': 'inference_dnn',
            'save_inference_feed_varnames': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                             '11', '12', '13', '14', '15', '16', '17', '18', '19',
                                             '20', '21', '22', '23', '24', '25', '26', 'dense_var'],
            'save_inference_fetch_varnames': ['fc_3.tmp_1'],
            'print_interval': 10,
            'phases': ['phase1']
        })
        self.yaml_content['runner'].append({
            'name': 'runner1',
            'class': 'train',
            'device': 'cpu',
            'epochs': 2,
            'init_model_path': 'increment_dnn/1',
            'print_interval': 10,
            'phases': ['phase1']
        })

        self.yaml_content['runner'].append({
            'name': 'runner2',
            'class': 'infer',
            'device': 'cpu',
            'epochs': 2,
            'init_model_path': 'increment_dnn/1',
            'print_interval': 10,
            'phases': ['phase2']
        })

    def _construct_phase(self):
        """_construct_phase of RankDNN"""
        self.yaml_content['phase'] = []
        self.yaml_content['phase'].append({
            'name': 'phase1',
            'model': '{workspace}/model.py',  # user-defined model
            'dataset_name': 'dataset_train',  # select dataset by name
            'thread_num': 1
        })
        self.yaml_content['phase'].append({
            'name': 'phase2',
            'model': '{workspace}/model.py',  # user-defined model
            'dataset_name': 'dataset_infer',  # select dataset by name
            'thread_num': 1
        })

    def _construct_dataset(self):
        """_construct_dataset of RankDNN"""
        self.yaml_content['dataset'] = []
        self.yaml_content['dataset'].append({
            'name': 'dataset_train',
            'dense_slots': 'dense_var:13',
            'sparse_slots': 'click 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26',
            'data_path': '{workspace}/data/sample_data/train',
            'type': 'DataLoader',  # 指定数据读取方式(DataLoader / QueueDataset)
            'batch_size': 2
            })
        self.yaml_content['dataset'].append({
            'dense_slots': 'dense_var:13',
            'name': 'dataset_infer',
            'sparse_slots': 'click 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26',
            'data_path': '{workspace}/data/sample_data/train',
            'type': 'DataLoader',
            'batch_size': 2})

    def _construct_hyper_paramters(self):
        """_construct_hyper_paramters of RankDNN"""
        self.yaml_content['hyper_parameters'] = {
            'sparse_feature_number': 1000001,
            'sparse_feature_dim': 9,
            'fc_sizes': [512, 256, 128, 64, 32],
            'optimizer': {
                'class': 'Adagrad',
                'learning_rate': 0.0001,
            }
        }
