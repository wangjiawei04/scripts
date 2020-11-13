#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file demo1.py
  * @author liyang109@baidu.com
  * @date 2020-05-18 11:05
  * @brief
  *
  **************************************************************************/
"""
import argparse
import ast
import copy
import logging
import multiprocessing
import os
import six
import sys
import time
import json
import numpy as np
import paddle.fluid as fluid
sys.path.append("./thirdparty/transformer")
from desc import *
from config import *
from transformer_args import parse_args
from model import transformer, position_encoding_init
from prepare_data import prepare_data_generator, prepare_feed_dict_list, py_reader_provider_wrapper
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.contrib.mixed_precision.decorator import decorate
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy

from dist_base_fleet import runtime_main
from dist_base_fleet import FleetDistRunnerBase
os.environ['FLAGS_sync_nccl_allreduce'] = "1"

args = parse_args()


class TestDistTrans(FleetDistRunnerBase):
    """Test Transformer fleet."""
    def net(self, args=None):
        """net struct."""
        pass

    def do_training(self, fleet, args):
        """
        begin training.
        Args:
            fleet (Collective): Collective inherited base class Fleet
            args (ArgumentParser): run args to config dist fleet.
        Returns:
            tuple: the value is train losses
        """
        args = parse_args()
        logging.info(args)
        gpu_id = int(os.environ.get('FLAGS_selected_gpus', 4))
        place = fluid.CUDAPlace(gpu_id)
        dev_count = 1
        exe = fluid.Executor(place)
        train_program = fluid.Program()
        startup_program = fluid.Program()
        args.num_trainers = fleet.worker_num()
        args.trainer_id = fleet.worker_index()
        args.run_params = json.loads(args.run_params)
        dist_strategy = DistributedStrategy()
        dist_strategy.enable_inplace = args.run_params['enable_inplace']
        dist_strategy.fuse_all_reduce_ops = args.run_params[
            'fuse_all_reduce_ops']
        dist_strategy.nccl_comm_num = args.run_params['nccl_comm_num']
        dist_strategy.use_local_sgd = args.run_params['use_local_sgd']
        dist_strategy.mode = args.run_params["mode"]
        dist_strategy.collective_mode = args.run_params["collective"]

        with fluid.program_guard(train_program, startup_program):
            with fluid.unique_name.guard():
                sum_cost, avg_cost, predict, token_num, pyreader = transformer(
                    ModelHyperParams.src_vocab_size,
                    ModelHyperParams.trg_vocab_size,
                    ModelHyperParams.max_length + 1,
                    ModelHyperParams.n_layer,
                    ModelHyperParams.n_head,
                    ModelHyperParams.d_key,
                    ModelHyperParams.d_value,
                    ModelHyperParams.d_model,
                    ModelHyperParams.d_inner_hid,
                    ModelHyperParams.prepostprocess_dropout,
                    ModelHyperParams.attention_dropout,
                    ModelHyperParams.relu_dropout,
                    ModelHyperParams.preprocess_cmd,
                    ModelHyperParams.postprocess_cmd,
                    ModelHyperParams.weight_sharing,
                    TrainTaskConfig.label_smooth_eps,
                    ModelHyperParams.bos_idx,
                    use_py_reader=args.use_py_reader,
                    is_test=False)
                optimizer = fluid.optimizer.SGD(0.003)
                if args.run_params["fp16"]:
                    optimizer = decorate(optimizer, init_loss_scaling=64.0)
                optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
                optimizer.minimize(avg_cost, startup_program)
        train_program = fleet.main_program
        exe.run(startup_program)
        train_data = prepare_data_generator(
            args,
            is_test=False,
            count=dev_count,
            pyreader=pyreader,
            py_reader_provider_wrapper=py_reader_provider_wrapper)

        loss_normalizer = -((1. - TrainTaskConfig.label_smooth_eps) * np.log(
            (1. - TrainTaskConfig.label_smooth_eps
             )) + TrainTaskConfig.label_smooth_eps *
                            np.log(TrainTaskConfig.label_smooth_eps / (
                                ModelHyperParams.trg_vocab_size - 1) + 1e-20))

        step_idx = 0
        init_flag = True
        result_loss = []
        result_ppl = []
        train_info = []
        for pass_id in six.moves.xrange(args.num_epochs):
            pass_start_time = time.time()
            if args.use_py_reader:
                pyreader.start()
                data_generator = None
            else:
                data_generator = train_data()
            batch_id = 0
            while True:
                try:
                    feed_dict_list = prepare_feed_dict_list(data_generator, init_flag, dev_count)
                    t1 = time.time()
                    outs = exe.run(program=train_program,
                                   fetch_list=[sum_cost.name, token_num.name]
                                   if step_idx % args.fetch_steps == 0 else [],
                                   feed=feed_dict_list)

                    if step_idx % args.fetch_steps == 0:
                        sum_cost_val, token_num_val = np.array(outs[0]), np.array(
                            outs[1])
                        total_sum_cost = sum_cost_val.sum()
                        total_token_num = token_num_val.sum()
                        total_avg_cost = total_sum_cost / total_token_num
                        result_loss.append(total_avg_cost - loss_normalizer)
                        result_ppl.append(np.exp([min(total_avg_cost, 100)]).item(0))
                        train_info.append(result_loss)
                    init_flag = False
                    batch_id += 1
                    step_idx += 1
                    if batch_id >= 5:
                        break
                except (StopIteration, fluid.core.EOFException):
                    if args.use_py_reader:
                        pyreader.reset()
                    break

            train_info = [round(i, 6) for i in train_info[0]]
            return train_info


if __name__ == "__main__":
    LOG_FORMAT = "[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(
        stream=sys.stdout, level=logging.DEBUG, format=LOG_FORMAT)
    logging.getLogger().setLevel(logging.INFO)
    runtime_main(TestDistTrans)