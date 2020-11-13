#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @Time    : 2019-09-12 13:43
# @Author  : liyang109

from __future__ import print_function

import paddle
import math
import time
import numpy as np
import paddle.fluid as fluid
import os
import sys
sys.path.append('./thirdparty/simnet_bow')
import py_reader_generator as py_reader1
# from cts_test.dist_fleet.reader_generator import ctr_py_reader_generator as py_reader1
from dist_base_fleet import runtime_main
from dist_base_fleet import FleetDistRunnerBase

params = {
    "is_first_trainer":True,
    "model_path":"dist_model_ctr",
    "is_pyreader_train":True,
    "is_dataset_train":False,
    "dict_dim":1451594,
    "emb_dim":128,
    "hid_dim":128,
    "learning_rate":0.2,
    "margin":0.1,
    "batch_size":128,
    "train_files_path":"./thirdparty/data/simnet_data/train_data",
    "is_local_cluster":True,
    "sample_rate":0.02,
    "cpu_num":2,
    "epochs":1
}

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1


class TestDistSimnetBow(FleetDistRunnerBase):
    """test dist simnet_bow"""
    def input_data(self):
        """input data"""
        with fluid.unique_name.guard():
            q = fluid.layers.data(
                name="query", shape=[1], dtype="int64", lod_level=1)
            pt = fluid.layers.data(
                name="pos_title", shape=[1], dtype="int64", lod_level=1)
            nt = fluid.layers.data(
                name="neg_title", shape=[1], dtype="int64", lod_level=1)

            self.inputs = [q, pt, nt]
        return self.inputs

    def py_reader(self):
        data_shapes = []
        data_lod_levels = []
        data_types = []
        # query ids
        data_shapes.append((-1, 1))
        data_lod_levels.append(1)
        data_types.append('int64')
        # pos_title_ids
        data_shapes.append((-1, 1))
        data_lod_levels.append(1)
        data_types.append('int64')
        # neg_title_ids
        data_shapes.append((-1, 1))
        data_lod_levels.append(1)
        data_types.append('int64')
        # label
        data_shapes.append((-1, 1))
        data_lod_levels.append(1)
        data_types.append('int64')

        self.reader = fluid.layers.py_reader(capacity=64,
                                        shapes=data_shapes,
                                        lod_levels=data_lod_levels,
                                        dtypes=data_types,
                                        name='py_reader',
                                        use_double_buffer=False)

        return self.reader

    def dataset_reader(self, inputs, params):
        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_batch_size(params["batch_size"])
        dataset.set_use_var([inputs[0], inputs[1], inputs[2]])
        dataset.set_batch_size(params["batch_size"])
        pipe_command = 'python dataset_generator.py'
        dataset.set_pipe_command(pipe_command)
        dataset.set_thread(int(params["cpu_num"]))
        return dataset

    def net(self, args=None):
        with fluid.unique_name.guard():
            is_distributed = False
            is_sparse = True
            inputs = self.input_data()
            self.pyreader = self.py_reader()
            inputs = fluid.layers.read_file(self.pyreader)
            query = inputs[0]
            pos = inputs[1]
            neg = inputs[2]
            dict_dim = params["dict_dim"]
            emb_dim = params["emb_dim"]
            hid_dim = params["hid_dim"]
            emb_lr = params["learning_rate"] * 3
            base_lr = params["learning_rate"]

            q_emb = fluid.layers.embedding(input=query,
                                           is_distributed=is_distributed,
                                           size=[dict_dim, emb_dim],
                                           param_attr=fluid.ParamAttr(name="__emb__",
                                                                      learning_rate=emb_lr,
                                                                      initializer=fluid.initializer.Xavier()),
                                           is_sparse=is_sparse
                                           )
            # vsum
            q_sum = fluid.layers.sequence_pool(
                input=q_emb,
                pool_type='sum')
            q_ss = fluid.layers.softsign(q_sum)
            # fc layer after conv
            q_fc = fluid.layers.fc(input=q_ss,
                                   size=hid_dim,
                                   param_attr=fluid.ParamAttr(name="__q_fc__", learning_rate=base_lr,
                                                              initializer=fluid.initializer.Xavier()))
            # embedding
            pt_emb = fluid.layers.embedding(input=pos,
                                            is_distributed=is_distributed,
                                            size=[dict_dim, emb_dim],
                                            param_attr=fluid.ParamAttr(name="__emb__", learning_rate=emb_lr,
                                                                       initializer=fluid.initializer.Xavier()),
                                            is_sparse=is_sparse)
            # vsum
            pt_sum = fluid.layers.sequence_pool(
                input=pt_emb,
                pool_type='sum')
            pt_ss = fluid.layers.softsign(pt_sum)
            # fc layer
            pt_fc = fluid.layers.fc(input=pt_ss,
                                    size=hid_dim,
                                    param_attr=fluid.ParamAttr(name="__fc__", learning_rate=base_lr,
                                                               initializer=fluid.initializer.Xavier()),
                                    bias_attr=fluid.ParamAttr(name="__fc_b__",
                                                              initializer=fluid.initializer.Xavier()))

            # embedding
            nt_emb = fluid.layers.embedding(input=neg,
                                            is_distributed=is_distributed,
                                            size=[dict_dim, emb_dim],
                                            param_attr=fluid.ParamAttr(name="__emb__",
                                                                       learning_rate=emb_lr,
                                                                       initializer=fluid.initializer.Xavier()),
                                            is_sparse=is_sparse)

            # vsum
            nt_sum = fluid.layers.sequence_pool(
                input=nt_emb,
                pool_type='sum')
            nt_ss = fluid.layers.softsign(nt_sum)
            # fc layer
            nt_fc = fluid.layers.fc(input=nt_ss,
                                    size=hid_dim,
                                    param_attr=fluid.ParamAttr(name="__fc__", learning_rate=base_lr,
                                                               initializer=fluid.initializer.Xavier()),
                                    bias_attr=fluid.ParamAttr(name="__fc_b__",
                                                              initializer=fluid.initializer.Xavier()))
            cos_q_pt = fluid.layers.cos_sim(q_fc, pt_fc)
            cos_q_nt = fluid.layers.cos_sim(q_fc, nt_fc)
            # loss
            loss_op1 = fluid.layers.elementwise_sub(
                fluid.layers.fill_constant_batch_size_like(input=cos_q_pt, shape=[-1, 1], value=params["margin"],
                                                           dtype='float32'), cos_q_pt)
            loss_op2 = fluid.layers.elementwise_add(loss_op1, cos_q_nt)
            loss_op3 = fluid.layers.elementwise_max(
                fluid.layers.fill_constant_batch_size_like(input=loss_op2, shape=[-1, 1], value=0.0,
                                                           dtype='float32'), loss_op2)
            self.avg_cost = fluid.layers.mean(loss_op3)
            # avg_cost = self.get_loss(cos_q_pt, cos_q_nt)
            # # acc
            # acc = self.get_acc(cos_q_nt, cos_q_pt)

        return self.avg_cost

    def do_training(self, fleet, args=None):
        exe = fluid.Executor(fluid.CPUPlace())
        fleet.init_worker()
        exe.run(fleet.startup_program)
        file_list = [str(params["train_files_path"]) + "/%s" % x
                     for x in os.listdir(params["train_files_path"])]
        # if params["is_local_cluster"]:
        # file_list = fleet.split_files(file_list)
        train_generator = py_reader1.get_batch_reader(file_list,
                                                     batch_size=params["batch_size"],
                                                     sample_rate=params["sample_rate"])
        self.pyreader.decorate_paddle_reader(train_generator)

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = int(params["cpu_num"])
        build_strategy = fluid.BuildStrategy()
    #    build_strategy.async_mode = self.async_mode
        if args.run_params["cpu_num"] > 1:
            build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
        compiled_prog = fluid.compiler.CompiledProgram(
            fleet.main_program).with_data_parallel(
            loss_name=self.avg_cost.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

        train_info = []
        for epoch in range(params["epochs"]):
            # Notice: py_reader should use try & catch EOFException method to enter the dataset
            # reader.start() must declare in advance
            self.pyreader.start()
            start_time = time.clock()
            batch_id = 0
            try:
                while True:
                    step_start = time.time()
                    avg_cost = exe.run(
                        program=compiled_prog,
                        fetch_list=[self.avg_cost.name])
                    avg_cost = np.mean(avg_cost)
                    step_end = time.time()
                    samples = params["batch_size"] * params["cpu_num"]

                    if batch_id % 10 == 0 and batch_id != 0:
                        print(
                            "Epoch: {0}, Step: {1}, Loss: {2}".format(
                                epoch, batch_id, avg_cost))
                        train_info.append(avg_cost)

                    batch_id += 1
                    if batch_id == 51:
                        break

            except fluid.core.EOFException:
                self.pyreader.reset()

            end_time = time.clock()
            fleet.stop_worker()
            if params["is_first_trainer"]:
                if params["is_pyreader_train"]:
                    model_path = str(params["model_path"] + "/final" + "_pyreader")
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
            if params["is_pyreader_train"]:
                model_path = params["model_path"] + "/final" + "_pyreader"
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
    runtime_main(TestDistSimnetBow)


