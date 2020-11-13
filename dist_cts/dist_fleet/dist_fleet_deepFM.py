#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2019-09-03 16:30
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

sys.path.append('./thirdparty/deepFM')
import py_reader_generator as py_reader1
from dist_base_fleet import runtime_main
from dist_base_fleet import FleetDistRunnerBase
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig

params = {
    "is_first_trainer": True,
    "model_path": "dist_model_deepFM",
    "is_pyreader_train": False,
    "is_dataset_train": False,
    "is_dataloader_train": True
}

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1
DATA_PATH = 'thirdparty/data/dist_data/ctr_data/part-100'

class TestDistDeepFM(FleetDistRunnerBase):
    """
    test distribute deepFM
    """
    def input_data(self):
        """input data"""
        num_field = 39
        raw_feat_idx = fluid.layers.data(
            name='feat_idx', shape=[num_field], dtype='int64')
        raw_feat_value = fluid.layers.data(
            name='feat_value', shape=[num_field], dtype='float32')
        label = fluid.layers.data(
            name='label', shape=[1], dtype='float32')  # None * 1
        self.inputs = [raw_feat_idx, raw_feat_value, label]
        return self.inputs

    def py_reader(self):
        """py_reader"""
        py_reader = fluid.layers.create_py_reader_by_data(capacity=64,
                                                          feed_list=self.inputs,
                                                          name='py_reader',
                                                          use_double_buffer=False)
        return py_reader

    def dataset_reader(self):
        """dataset reader"""
        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_use_var(self.inputs)
        pipe_command = "python dataset_generator.py"
        dataset.set_pipe_command(pipe_command)
        dataset.set_batch_size(params.batch_size)
        thread_num = int(params.cpu_num)
        dataset.set_thread(thread_num)
        return dataset

    def net(self, args=None):
        """net structure"""
        inputs = self.input_data()
        self.loader = fluid.io.DataLoader.from_generator(capacity=64,
                                                         use_double_buffer=True,
                                                         feed_list=self.inputs,
                                                         iterable=False)

        init_value_ = 0.1
        raw_feat_idx = inputs[0]
        raw_feat_value = inputs[1]
        label = inputs[2]
        feat_idx = fluid.layers.reshape(raw_feat_idx,
                                        [-1, 39, 1])
        feat_value = fluid.layers.reshape(
            raw_feat_value, [-1, 39, 1])

        first_weights = fluid.layers.embedding(
            input=feat_idx,
            dtype='float32',
            size=[1086460 + 1, 1],
            padding_idx=0,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=init_value_),
                regularizer=fluid.regularizer.L1DecayRegularizer(
                    1e-4)))
        y_first_order = fluid.layers.reduce_sum((first_weights * feat_value), 1)
        feat_embeddings = fluid.layers.embedding(
            input=feat_idx,
            dtype='float32',
            size=[1086460 + 1, 10],
            padding_idx=0,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=init_value_ / math.sqrt(float(10)))))
        feat_embeddings = feat_embeddings * feat_value
        summed_features_emb = fluid.layers.reduce_sum(feat_embeddings, 1)
        summed_features_emb_square = fluid.layers.square(summed_features_emb)
        squared_features_emb = fluid.layers.square(feat_embeddings)
        squared_sum_features_emb = fluid.layers.reduce_sum(squared_features_emb, 1)
        y_second_order = 0.5 * fluid.layers.reduce_sum(
            summed_features_emb_square - squared_sum_features_emb, 1, keep_dim=True)
        y_dnn = fluid.layers.reshape(feat_embeddings, [-1, 39 * 10])
        for s in [400, 400, 400]:
            y_dnn = fluid.layers.fc(
                input=y_dnn,
                size=s,
                act='relu',
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=init_value_ / math.sqrt(float(10)))),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=init_value_)))
        y_dnn = fluid.layers.fc(
            input=y_dnn,
            size=1,
            act=None,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=init_value_)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=init_value_)))
        predict = fluid.layers.sigmoid(y_first_order + y_second_order + y_dnn)
        cost = fluid.layers.log_loss(input=predict, label=label)
        self.avg_cost = fluid.layers.reduce_sum(cost)
        predict_2d = fluid.layers.concat([1 - predict, predict], 1)
        label_int = fluid.layers.cast(label, 'int64')
        auc_var, batch_auc_var, auc_states = fluid.layers.auc(input=predict_2d,
                                                              label=label_int,
                                                              slide_steps=0)
        return self.avg_cost

    def check_model_right(self, dirname):
        """check model"""
        model_filename = os.path.join(dirname, "__model__")
        with open(model_filename, "rb") as f:
            program_desc_str = f.read()
        program = fluid.Program.parse_from_string(program_desc_str)
        with open(os.path.join(dirname, "__model__.proto"), "w") as wn:
            wn.write(str(program))

    def do_training(self, fleet, args=None):
        exe = fluid.Executor(fluid.CPUPlace())
        fleet.init_worker()
        exe.run(fleet.startup_program)
        train_generator = py_reader1.CriteoDataset()
        file_list = [str(DATA_PATH)] * 2 
        # file_list = fleet.split_files(file_list)
        train_reader = paddle.batch(
            train_generator.train(file_list),
            batch_size=4)
        self.loader.set_sample_list_generator(train_reader)
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = int(2)
        build_strategy = fluid.BuildStrategy()
        if args.run_params['cpu_num'] > 1:
            build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
        compiled_prog = fluid.compiler.CompiledProgram(
            fleet.main_program).with_data_parallel(
            loss_name=self.avg_cost.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

        for epoch in range(1):
            self.loader.start()
            start_time = time.clock()
            train_info = []
            batch_id = 0
            try:
                while True:
                    loss_val = exe.run(
                        program=compiled_prog,
                        fetch_list=[self.avg_cost.name])
                    loss_val = np.mean(loss_val)
                    train_info.append(loss_val)
                    if batch_id % 10 == 0 and batch_id != 0:
                        print(
                            "TRAIN --> pass: {} batch: {} loss: {}"
                                .format(epoch, batch_id, loss_val /
                                        100))
                    batch_id += 1
                    if batch_id == 5:
                        break
            except fluid.core.EOFException:
                self.loader.reset()
            end_time = time.clock()
        fleet.stop_worker()
        if params["is_first_trainer"]:
            if params["is_dataloader_train"]:
                model_path = str(params["model_path"] + "/final" + "_dataloader")
                fleet.save_persistables(
                    executor=fluid.Executor(fluid.CPUPlace()),
                    dirname=model_path)
            elif params["is_dataset_train"]:
                model_path = str(params["model_path"] + '/final' + "_dataset")
                fleet.save_persistables(
                    executor=fluid.Executor(fluid.CPUPlace()),
                    dirname=model_path)
            else:
                raise ValueError("Program must has Date feed method: is_pyreader_train / is_dataset_train")
        if params["is_dataloader_train"]:
            model_path = params["model_path"] + "/final" + "_dataloader"
            fluid.io.load_persistables(
                executor=fluid.Executor(fluid.CPUPlace()),
                dirname=model_path,
                main_program=fluid.default_main_program())
        elif params["is_dataset_train"]:
            model_path = params["model_path"] + "/final" + "_dataset"
            fluid.io.load_persistables(
                executor=fluid.Executor(fluid.CPUPlace()),
                dirname=model_path,
                main_program=fluid.default_main_program())
        else:
            raise ValueError("No such model path")
        return train_info


if __name__ == "__main__":
    runtime_main(TestDistDeepFM)
