#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
#======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
#======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file dist_fleet_vgg.py
  * @author liyang109@baidu.com
  * @date 2020-11-05 15:53
  * @brief
  *
  **************************************************************************/
"""
from __future__ import print_function
import paddle.fluid as fluid
import os
import sys
import paddle
import json
import numpy as np
from dist_base_fleet import runtime_main
from dist_base_fleet import FleetDistRunnerBase
import thirdparty.image_classfication.utils.reader_cv2 as reader
sys.path.append("./thirdparty/image_classfication")


# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1
np.random.seed(1)
DATA_DIR = './thirdparty/data/dist_data/ImageNet'


class TestDistVgg16(FleetDistRunnerBase):
    """Test Vgg16 fleet."""

    def __init__(self):
        FleetDistRunnerBase.__init__(self, batch_num=5, batch_size=1000)
        self.model_name = 'VGG16'
        self.batch_size = 8
        self.batch_num = 5

    def linear_warmup_decay(self, learning_rate, warmup_steps, num_train_steps):
        """ Applies linear warmup of learning rate from 0 and decay to 0."""
        with fluid.default_main_program()._lr_schedule_guard():
            lr = fluid.layers.tensor.create_global_var(
                shape=[1],
                value=0.0,
                dtype='float32',
                persistable=True,
                name="scheduled_learning_rate")

            global_step = fluid.layers.learning_rate_scheduler._decay_step_counter(
            )

            with fluid.layers.control_flow.Switch() as switch:
                with switch.case(global_step < warmup_steps):
                    warmup_lr = learning_rate * (global_step / warmup_steps)
                    fluid.layers.tensor.assign(warmup_lr, lr)
                with switch.default():
                    decayed_lr = fluid.layers.learning_rate_scheduler.polynomial_decay(
                        learning_rate=learning_rate,
                        decay_steps=num_train_steps,
                        end_learning_rate=0.0,
                        power=1.0,
                        cycle=False)
                    fluid.layers.tensor.assign(decayed_lr, lr)

            return lr

    def net(self, args=None):
        """
        vgg net struct.
        Args:
            fleet:
            args (ArgumentParser): run args to config dist fleet.
        Returns:
            tuple: the return value contains avg_cost, py_reader
        """
        import paddle.distributed.fleet as fleet
        # from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
        from thirdparty.image_classfication.models.vgg import VGG16
        from thirdparty.image_classfication.train import parser
        from thirdparty.image_classfication.train import optimizer_setting
        parser.add_argument(
            '--update_method',
            type=str,
            required=True,
            choices=['pserver', 'nccl'])
        parser.add_argument(
            '--role', type=str, required=True, choices=['pserver', 'trainer'])
        parser.add_argument(
            '--endpoints', type=str, required=False, default="")
        parser.add_argument(
            '--current_id', type=int, required=False, default=0)
        parser.add_argument('--trainers', type=int, required=False, default=1)
        # parser.add_argument('--sync_mode', action='store_true')
        parser.add_argument(
            '--run_params', type=str, required=False, default='{}')
        args = parser.parse_args()
        args.run_params = json.loads(args.run_params)
        print(args.run_params)

        image_shape = [3, 224, 224]
        scale_loss = 1.0
        fluid.io.reader.keep_data_loader_order(False)
        image = fluid.data(shape=[-1, 3, 224, 224], dtype='float32', name='image')
        label = fluid.data(shape=[-1, 1], dtype='int64', name='label')
        self.loader = fluid.io.DataLoader.from_generator(capacity=16,
                                                         use_double_buffer=True,
                                                         feed_list=[image, label],
                                                         iterable=False)

        run_model = VGG16()
        out = run_model.net(image, 4)
        softmax_out = fluid.layers.softmax(out, use_cudnn=False)
        cost, prob = fluid.layers.softmax_with_cross_entropy(
            out, label, return_softmax=True)
        self.avg_cost = fluid.layers.mean(cost)
        checkpoints = run_model.checkpoints
        params = run_model.params
        params["total_images"] = args.total_images
        params["lr"] = 1e-5
        params["num_epochs"] = args.num_epochs
        params["learning_strategy"]["batch_size"] = args.batch_size
        params["learning_strategy"]["name"] = args.lr_strategy
        params["l2_decay"] = args.l2_decay
        params["momentum_rate"] = args.momentum_rate
        optimizer = optimizer_setting(params)
        global_lr = optimizer._global_learning_rate()
        global_lr.persistable = True

        build_strategy = paddle.fluid.BuildStrategy()
        build_strategy.enable_inplace = args.run_params['enable_inplace']

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = 1
        exec_strategy.num_iteration_per_drop_scope = 3

        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.execution_strategy = exec_strategy
        dist_strategy.dgc = args.run_params['dgc']
        dist_strategy.lars = args.run_params['lars']
        dist_strategy.lamb = args.run_params['lamb']
        dist_strategy.auto = args.run_params['auto']
        dist_strategy.sync_batch_norm = args.run_params['sync_bn']
        dist_strategy.nccl_comm_num = args.run_params['nccl_comm_num']
        dist_strategy.fuse_all_reduce_ops = args.run_params['fuse_all_reduce_ops']
        dist_strategy.sync_nccl_allreduce = args.run_params['sync_nccl_allreduce']
        dist_strategy.hierarchical_allreduce_inter_nranks = args.run_params["inter_nranks"]

        if bool(args.run_params['recompute']):
            dist_strategy.recompute = True
            dist_strategy.recompute_configs = {"checkpoints": ["fc_1.tmp_1", "fc_1.tmp_2"]}
        else:
            dist_strategy.recompute = False

        if bool(args.run_params["pipeline"]):
            dist_strategy.pipeline = True
            dist_strategy.pipeline_configs = {"micro_batch": 2}
        else:
            dist_strategy.pipeline = False

        if bool(args.run_params["localsgd"]):
            dist_strategy.localsgd = True
            dist_strategy.localsgd_configs = {"k_steps": 2}
        else:
            dist_strategy.localsgd = False

        if bool(args.run_params['gradient_merge']):
            dist_strategy.gradient_merge = True
            dist_strategy.gradient_merge_configs = {"k_steps": 2}
        else:
            dist_strategy.gradient_merge = False


        if args.run_params["fp16"]:
            optimizer = fluid.contrib.mixed_precision.decorate(
                optimizer,
                init_loss_scaling=128.0,
                use_dynamic_loss_scaling=True)
        if args.run_params["gradient_merge"]:
            scheduled_lr = self.linear_warmup_decay(learning_rate=0.01, warmup_steps=16000,
                                               num_train_steps=1000000)
            optimizer = fluid.optimizer.Adam(learning_rate=scheduled_lr)

        if "use_dgc" in args.run_params and args.run_params["use_dgc"]:
            # use dgc must close fuse
            dist_strategy.fuse_all_reduce_ops = False
            optimizer = fluid.optimizer.DGCMomentumOptimizer(
                learning_rate=0.001, momentum=0.9, rampup_begin_step=0)

        dist_optimizer = fleet.distributed_optimizer(
            optimizer, strategy=dist_strategy)
        _, param_grads = dist_optimizer.minimize(self.avg_cost)

        shuffle_seed = 1
        train_reader = reader.train(
            settings=args, data_dir=DATA_DIR, pass_id_as_seed=shuffle_seed)

        self.loader.set_sample_list_generator(paddle.batch(
                train_reader, batch_size=self.batch_size))

        if scale_loss > 1:
            avg_cost = fluid.layers.mean(x=cost) * scale_loss
        return self.avg_cost, self.loader

    def check_model_right(self, dirname):
        """
        check model.
        Args:
            dirname (str): model saved dirname.
        """
        model_filename = os.path.join(dirname, "__model__")
        with open(model_filename, "rb") as f:
            program_desc_str = f.read()
        program = fluid.Program.parse_from_string(program_desc_str)
        with open(os.path.join(dirname, "__model__.proto"), "w") as wn:
            wn.write(str(program))

    def py_reader(self):
        """rewrite py_reader"""
        pass

    def dataset_reader(self):
        """rewriter dataset_reader"""
        pass

    def do_training(self, fleet, args=None):
        """
        begin training.
        Args:
            fleet (Collective): Collective inherited base class Fleet
            args (ArgumentParser): run args to config dist fleet.
        Returns:
            tuple: the value is train losses
        """
        train_fetch_vars = [self.avg_cost]
        train_fetch_list = []
        for var in train_fetch_vars:
            var.persistable = True
            train_fetch_list.append(var.name)

        trainer_prog = fluid.default_main_program()
        dist_prog = fluid.default_main_program()

        device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
        place = fluid.CUDAPlace(device_id)

        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        train_info = []
        for pass_id in range(1):
            self.loader.start()
            batch_id = 0
            try:
                while True:
                    loss, = exe.run(dist_prog, fetch_list=train_fetch_list)
                    loss = np.mean(np.array(loss))
                    train_info.append(loss)
                    batch_id += 1
                    if batch_id == 5:
                        break
            except fluid.core.EOFException:
                self.loader.reset()
        return train_info


if __name__ == "__main__":
    paddle.enable_static()
    runtime_main(TestDistVgg16)
