#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @Time    : 2019-12-17 11:23
# @Author  : liyang109
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import math
import shutil
import tempfile
import time
import numpy as np
import signal
import paddle.fluid as fluid
import os
import argparse
import sys
from dist_base_fleet import runtime_main
from dist_base_fleet import FleetDistRunnerBase
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.WARNING)

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1
params = {'train_files_path': "./thirdparty/data/word2vec/dataset/train_data/"}


class TestDistDataSet(FleetDistRunnerBase):
    """dataset reader data"""
    def input_data(self):
         with fluid.unique_name.guard():
             input_word = fluid.layers.data(name="input_word", shape=[
                 1], dtype='int64', lod_level=1)
             true_word = fluid.layers.data(name='true_label', shape=[
                 1], dtype='int64', lod_level=1)
             neg_word = fluid.layers.data(
                 name="neg_label", shape=[1], dtype='int64', lod_level=1)
             inputs = [input_word, true_word, neg_word]
         return self.inputs

    def dataset_reader(self):
        """dataset reader"""
        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_use_var(self.inputs)
        pipe_command = "python dataset_generator.py"
        dataset.set_pipe_command(pipe_command)
        dataset.set_batch_size(100)
        thread_num = 2
        dataset.set_thread(thread_num)
        return dataset

    def net(self, args=None):
        """word2vec net"""
        with fluid.unique_name.guard():
            input_word = fluid.layers.data(name="input_word", shape=[
                1], dtype='int64', lod_level=1)
            true_word = fluid.layers.data(name='true_label', shape=[
                1], dtype='int64', lod_level=1)
            neg_word = fluid.layers.data(
                name="neg_label", shape=[1], dtype='int64', lod_level=1)
            self.inputs = [input_word, true_word, neg_word]

            init_width = 0.5 / 300
            input_emb = fluid.layers.embedding(
                input=self.inputs[0],
                is_sparse=True,
                size=[354051, 300],
                param_attr=fluid.ParamAttr(
                    name='emb',
                    initializer=fluid.initializer.Uniform(-init_width, init_width)))

            true_emb_w = fluid.layers.embedding(
                input=self.inputs[1],
                is_sparse=True,
                size=[354051, 300],
                param_attr=fluid.ParamAttr(
                    name='emb_w', initializer=fluid.initializer.Constant(value=0.0)))

            true_emb_b = fluid.layers.embedding(
                input=self.inputs[1],
                is_sparse=True,
                size=[354051, 1],
                param_attr=fluid.ParamAttr(
                    name='emb_b', initializer=fluid.initializer.Constant(value=0.0)))

            neg_word_reshape = fluid.layers.reshape(self.inputs[2], shape=[-1, 1])
            neg_word_reshape.stop_gradient = True

            neg_emb_w = fluid.layers.embedding(
                input=neg_word_reshape,
                is_sparse=True,
                size=[354051, 300],
                param_attr=fluid.ParamAttr(
                    name='emb_w', learning_rate=1.0))

            neg_emb_w_re = fluid.layers.reshape(
                neg_emb_w, shape=[-1, 5, 300])

            neg_emb_b = fluid.layers.embedding(
                input=neg_word_reshape,
                is_sparse=True,
                size=[354051, 1],
                param_attr=fluid.ParamAttr(
                    name='emb_b', learning_rate=1.0))

            neg_emb_b_vec = fluid.layers.reshape(neg_emb_b, shape=[-1, 5])

            true_logits = fluid.layers.elementwise_add(
                fluid.layers.reduce_sum(
                    fluid.layers.elementwise_mul(input_emb, true_emb_w),
                    dim=1,
                    keep_dim=True),
                true_emb_b)

            input_emb_re = fluid.layers.reshape(
                input_emb, shape=[-1, 1, 300])

            neg_matmul = fluid.layers.matmul(
                input_emb_re, neg_emb_w_re, transpose_y=True)
            neg_matmul_re = fluid.layers.reshape(neg_matmul, shape=[-1, 5])
            neg_logits = fluid.layers.elementwise_add(neg_matmul_re, neg_emb_b_vec)
            # nce loss
            label_ones = fluid.layers.fill_constant_batch_size_like(
                true_logits, shape=[-1, 1], value=1.0, dtype='float32')
            label_zeros = fluid.layers.fill_constant_batch_size_like(
                true_logits, shape=[-1, 5], value=0.0, dtype='float32')

            true_xent = fluid.layers.sigmoid_cross_entropy_with_logits(true_logits,
                                                                       label_ones)
            neg_xent = fluid.layers.sigmoid_cross_entropy_with_logits(neg_logits,
                                                                      label_zeros)
            cost = fluid.layers.elementwise_add(
                fluid.layers.reduce_sum(
                    true_xent, dim=1),
                fluid.layers.reduce_sum(
                    neg_xent, dim=1))
            self.avg_cost = fluid.layers.reduce_mean(cost)
        return self.avg_cost

    def do_training(self, fleet, args=None):
        """training"""
        exe = fluid.Executor(fluid.CPUPlace())
        fleet.init_worker()
        exe.run(fleet.startup_program)
        train_info = []
        dataset = self.dataset_reader()
        file_list = [str(params['train_files_path']) + "/%s" % x for x in os.listdir(params['train_files_path'])]
        file_list = fleet.split_files(file_list)
        for epoch in range(1):
            dataset.set_filelist(file_list)
            start_time = time.time()
            var_dict = {"loss": self.avg_cost}
            global var_dict
            class FetchVars(fluid.executor.FetchHandler):
                def __init__(self, var_dict=None, period_secs=5):
                    super(FetchVars, self).__init__(var_dict, period_secs=5)
                def handler(self, res_dict):
                   # with open('out.txt', 'w+') as file:
                     #   sys.stdout = file
                    train_info.extend(res_dict['loss'].tolist())
                    print(train_info)
            exe.train_from_dataset(program=fleet.main_program,
                                   dataset=dataset,
                                   fetch_handler=FetchVars(var_dict))
            end_time = time.time()
            all_time = end_time - start_time
        logger.info("Train Success!")
        fleet.stop_worker()
        return train_info


if __name__ == '__main__':
    runtime_main(TestDistDataSet)
