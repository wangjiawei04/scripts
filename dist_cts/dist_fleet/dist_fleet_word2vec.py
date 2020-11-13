#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @Time    : 2019-12-02 16:42
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
import paddle.fluid as fluid
import os
import argparse
import sys
sys.path.append('./thirdparty/word2vec')
import py_reader_generator as py_reader1
from dist_base_fleet import runtime_main
from dist_base_fleet import FleetDistRunnerBase
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig


# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1

dict_params = {
    "embedding_size": 64,
    "is_sparse": False,
    "dict_size": 63642,
    "neg_num": 5,
    "train_files_path": "./data/word2vec/train_data",
    "dict_path": "./data/word2vec/test_build_dict",
    "batch_size": 500,
    "is_local_cluster": "True",
    "epochs": 1,
}

class TestDistWord2Vec(FleetDistRunnerBase):
    """dist word2vec"""
    def input_data(self):
        """input data"""
        with fluid.unique_name.guard():
            input_word = fluid.layers.data(name="input_word", shape=[1], dtype='int64',lod_level=1)
            true_word = fluid.layers.data(name='true_label', shape=[1], dtype='int64',lod_level=1)
            neg_word = fluid.layers.data(
                name="neg_label", shape=[1], dtype='int64',lod_level=1)
            self.inputs = [input_word, true_word, neg_word]
        return self.inputs

    def py_reader(self):
        """pyreader"""
        self.reader = fluid.layers.create_py_reader_by_data(
            capacity=64,
            feed_list=self.inputs,
            name='reader',
            use_double_buffer=False)
        return self.reader

    def dataset_reader(self, input, params):
        """dataset reader"""
        pass

    def net(self, args=None):
        """net struct"""
        init_width = 0.5 / dict_params["embedding_size"]
        inputs = self.input_data()
        self.reader = self.py_reader()
        inputs = fluid.layers.read_file(self.reader)
        input_emb = fluid.layers.embedding(
            input=inputs[0],
            is_sparse=dict_params["is_sparse"],
            size=[dict_params["dict_size"], dict_params["embedding_size"]],
            param_attr=fluid.ParamAttr(
                name='emb',
                initializer=fluid.initializer.Uniform(-init_width, init_width)))

        true_emb_w = fluid.layers.embedding(
            input=inputs[1],
            is_sparse=dict_params["is_sparse"],
            size=[dict_params["dict_size"], dict_params["embedding_size"]],
            param_attr=fluid.ParamAttr(
                name='emb_w',
                initializer=fluid.initializer.Constant(value=0.0)))

        true_emb_b = fluid.layers.embedding(
            input=inputs[1],
            is_sparse=dict_params["is_sparse"],
            size=[dict_params["dict_size"], 1],
            param_attr=fluid.ParamAttr(
                name='emb_b',
                initializer=fluid.initializer.Constant(value=0.0)))

        neg_word_reshape = fluid.layers.reshape(inputs[2], shape=[-1, 1])
        neg_word_reshape.stop_gradient = True

        neg_emb_w = fluid.layers.embedding(
            input=neg_word_reshape,
            is_sparse=dict_params["is_sparse"],
            size=[dict_params["dict_size"], dict_params["embedding_size"]],
            param_attr=fluid.ParamAttr(
                name='emb_w',
                learning_rate=1.0))

        neg_emb_w_re = fluid.layers.reshape(
            neg_emb_w, shape=[-1, dict_params["neg_num"], dict_params["embedding_size"]])

        neg_emb_b = fluid.layers.embedding(
            input=neg_word_reshape,
            is_sparse=dict_params["is_sparse"],
            size=[dict_params["dict_size"], 1],
            param_attr=fluid.ParamAttr(
                name='emb_b', learning_rate=1.0))

        neg_emb_b_vec = fluid.layers.reshape(neg_emb_b, shape=[-1, dict_params["neg_num"]])

        true_logits = fluid.layers.elementwise_add(
            fluid.layers.reduce_sum(
                fluid.layers.elementwise_mul(input_emb, true_emb_w),
                dim=1,
                keep_dim=True),
            true_emb_b)

        input_emb_re = fluid.layers.reshape(
            input_emb, shape=[-1, 1, dict_params["embedding_size"]])

        neg_matmul = fluid.layers.matmul(
            input_emb_re, neg_emb_w_re, transpose_y=True)
        neg_matmul_re = fluid.layers.reshape(neg_matmul, shape=[-1, dict_params["neg_num"]])
        neg_logits = fluid.layers.elementwise_add(neg_matmul_re, neg_emb_b_vec)
        # nce loss
        label_ones = fluid.layers.fill_constant_batch_size_like(
            true_logits, shape=[-1, 1], value=1.0, dtype='float32')
        label_zeros = fluid.layers.fill_constant_batch_size_like(
            true_logits, shape=[-1, dict_params["neg_num"]], value=0.0, dtype='float32')

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

    def check_model_right(self, dirname):
            """check """
            model_filename = os.path.join(dirname, "__model__")
            with open(model_filename, "rb") as f:
                program_desc_str = f.read()
            program = fluid.Program.parse_from_string(program_desc_str)
            with open(os.path.join(dirname, "__model__.proto"), "w") as wn:
                wn.write(str(program))

    def do_training(self, fleet, args=None):
        """training"""
        exe = fluid.Executor(fluid.CPUPlace())
        fleet.init_worker()

        exe.run(fleet.startup_program)
        CPU_NUM = 2
        file_list = os.listdir(dict_params["train_files_path"])
        file_list = fleet.split_files(file_list)

        word2vec_reader = py_reader1.Word2VecReader(dict_params["dict_path"],
                                                    dict_params["train_files_path"],
                                                    file_list, 0, 1)
        np_power = np.power(np.array(word2vec_reader.id_frequencys), 0.75)
        id_frequencys_pow = np_power / np_power.sum()
   #     self.reader.decorate_paddle_reader(
   #         py_reader1.convert_python_to_tensor(id_frequencys_pow, dict_params["batch_size"], word2vec_reader.train()))
        self.reader.decorate_tensor_provider(
            py_reader1.convert_python_to_tensor(id_frequencys_pow, dict_params["batch_size"], word2vec_reader.train()))

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = 2
        exec_strategy.use_experimental_executor = True
        build_strategy = fluid.BuildStrategy()

        if CPU_NUM > 1:
            build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce

        compiled_prog = fluid.compiler.CompiledProgram(
            fleet.main_program).with_data_parallel(
            loss_name=self.avg_cost.name, build_strategy=build_strategy, exec_strategy=exec_strategy)

        train_info = []
        for epoch in range(1):
            self.reader.start()
            start_time = time.clock()
            batch_id = 0
            try:
                while True:
                    step_start = time.time()
                    avg_cost = exe.run(
                        program=compiled_prog,
                        fetch_list=[self.avg_cost])
                    avg_cost = np.mean(avg_cost)
                    step_end = time.time()
                    samples = dict_params["batch_size"] * 2
                    if batch_id % 10 == 0 and batch_id != 0:
                        print(
                            "Epoch: {0}, Step: {1}, Loss: {2}"
                                .format(epoch, batch_id, avg_cost))
                        train_info.append(avg_cost)
                    batch_id += 1
                    if batch_id == 51:
                        break
            except fluid.core.EOFException:
                self.reader.reset()
            end_time = time.clock()
            cost_time = end_time - start_time
            fleet.stop_worker()
        return train_info

if __name__ == '__main__':
    runtime_main(TestDistWord2Vec)
