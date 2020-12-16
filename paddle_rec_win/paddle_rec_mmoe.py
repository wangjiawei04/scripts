#!/usr/bin/env python
# -*- coding: utf-8 -*-

from paddle_rec_base import PaddleRecBase

class MultiTaskMMOEBase(PaddleRecBase):
    """RankDNN test"""
    def __init__(self):
        super(MultiTaskMMOEBase, self).__init__()

    def _construct_global_vars(self):
        self.yaml_content['debug'] = True
        self.yaml_content['workspace'] = "models/multitask/mmoe"
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
            'save_checkpoint_path': 'increment_mmoe',
            'save_inference_interval': 2,
            'save_inference_path': 'inference_mmoe',
            # 'save_inference_feed_varnames': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
            #                                  '11', '12', '13', '14', '15', '16', '17', '18', '19',
            #                                  '20', '21', '22', '23', '24', '25', '26', 'dense_var'],
            # 'save_inference_fetch_varnames': ['fc_3.tmp_1'],
            'print_interval': 10,
        })
        # 增量训练
        self.yaml_content['runner'].append({
            'name': 'runner1',
            'class': 'train',
            'device': 'cpu',
            'epochs': 2,
            'init_model_path': 'increment_mmoe/1'
        })
        # test
        self.yaml_content['runner'].append({
            'name': 'runner2',
            'class': 'infer',
            'device': 'cpu',
            'epochs': 2,
            'init_model_path': 'increment_mmoe/1'
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
            'batch_size': 1,
            'type': 'DataLoader',
            'data_path': "{workspace}/data/train",
            'data_converter': "{workspace}/census_reader.py"
            })
        self.yaml_content['dataset'].append({
            'name': 'dataset_infer',
            'batch_size': 1,
            'type': 'DataLoader',
            'data_path': "{workspace}/data/train",
            'data_converter': "{workspace}/census_reader.py"
            })

    def _construct_hyper_paramters(self):
        """_construct_hyper_paramters of RankDNN"""
        self.yaml_content['hyper_parameters'] = {
            'feature_size': 499,
            'expert_num': 8,
            'gate_num': 2,
            'expert_size': 16,
            'tower_size': 8,
            'optimizer': {
                 'class': 'adam',
                 'learning_rate': 0.001,
                 'strategy': 'async'
            }
        }
