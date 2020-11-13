#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file dist_bert.py
  * @author liyang109@baidu.com
  * @date 2020-05-18 15:53
  * @brief
  *
  **************************************************************************/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import time
import paddle
import argparse
import numpy as np
import subprocess
import multiprocessing
import paddle.fluid as fluid
sys.path.append("./thirdparty/bert")
import reader.cls as reader
from model.bert import BertConfig
from model.classifier import create_model
from optimization import optimization
from utils.args import ArgumentGroup, print_arguments, check_cuda
from utils.init import init_pretraining_params, init_checkpoint
from utils.cards import get_cards
from env import dist_env
from bert_args import p_args
import dist_utils
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from dist_base_fleet import runtime_main
from dist_base_fleet import FleetDistRunnerBase

from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import paddle.fluid.incubate.fleet.base.role_maker as role_maker


DATA_DIR = './thirdparty/data/dist_data/bert_data/'
num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))

class TestDistBert(FleetDistRunnerBase):
    """Test BERT fleet."""
    def net(self, args=None):
        """
        BERT net struct.
        Args:
            fleet:
            args (ArgumentParser): run args to config dist fleet.
        Returns:
            tuple: the return value contains avg_cost, py_reader
        """
        args = p_args()
        bert_config = BertConfig(DATA_DIR + "uncased_L-24_H-1024_A-16/bert_config.json")
        bert_config.print_config()
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        exe = fluid.Executor(place)
        # init program
        train_program = fluid.Program()
        startup_prog = fluid.Program()

        if args.random_seed != 0:
            print("set program random seed as: ", args.random_seed)
            startup_prog.random_seed = args.random_seed
            train_program.random_seed = args.random_seed

        task_name = args.task_name.lower()
        processors = {
            'xnli': reader.XnliProcessor,
            'cola': reader.ColaProcessor,
            'mrpc': reader.MrpcProcessor,
            'mnli': reader.MnliProcessor,
        }
        processor = processors[task_name](data_dir=args.data_dir,
                                          vocab_path=args.vocab_path,
                                          max_seq_len=args.max_seq_len,
                                          do_lower_case=args.do_lower_case,
                                          in_tokens=args.in_tokens,
                                          random_seed=args.random_seed)
        num_labels = len(processor.get_labels())

        dev_count = 1
        self.train_data_generator = processor.data_generator(
            batch_size=args.batch_size,
            phase='train',
            epoch=args.epoch,
            dev_count=dev_count,
            dev_idx=0,
            shuffle=args.shuffle,
            shuffle_seed=args.shuffle_seed)

        num_train_examples = processor.get_num_examples(phase='train')

        max_train_steps = 5
        self.warmup_steps = 0.5

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.use_experimental_executor = args.use_fast_executor
        exec_strategy.num_threads = dev_count
        exec_strategy.num_iteration_per_drop_scope = args.num_iteration_per_drop_scope

        dist_strategy = DistributedStrategy()
        args.run_params = json.loads(args.run_params)
        dist_strategy.enable_inplace = args.run_params['enable_inplace']
        dist_strategy.fuse_all_reduce_ops = args.run_params[
            'fuse_all_reduce_ops']
        dist_strategy.nccl_comm_num = args.run_params['nccl_comm_num']
        dist_strategy.use_local_sgd = args.run_params['use_local_sgd']
        dist_strategy.mode = args.run_params["mode"]
        dist_strategy.collective_mode = args.run_params["collective"]
        dist_strategy.exec_strategy = exec_strategy
        dist_strategy.use_hierarchical_allreduce = False

        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                self.train_pyreader, self.loss, probs, accuracy, num_seqs, checkpoints = create_model(
                    args,
                    bert_config=bert_config,
                    num_labels=num_labels)
                scheduled_lr = optimization(
                    loss=self.loss,
                    warmup_steps=self.warmup_steps,
                    num_train_steps=max_train_steps,
                    learning_rate=args.learning_rate,
                    train_program=train_program,
                    startup_prog=startup_prog,
                    weight_decay=args.weight_decay,
                    scheduler=args.lr_scheduler,
                    use_fp16=False,
                    loss_scaling=args.loss_scaling,
                    dist_strategy=dist_strategy)
        exe.run(startup_prog)
        with open("__model__", "wb") as f:
            f.write(fleet._origin_program.desc.serialize_to_string())

        with open("debug_program", "w") as f:
            f.write(str(fleet._origin_program))
        return self.loss

    def do_training(self, fleet, args=None):
        """
        begin training.
        Args:
            fleet (Collective): Collective inherited base class Fleet
            args (ArgumentParser): run args to config dist fleet.
        Returns:
            tuple: the value is train losses
        """
        args = p_args()
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        exe = fluid.Executor(place)
        self.train_pyreader.decorate_batch_generator(self.train_data_generator, place)
        steps = 0
        train_info = []
        for data in self.train_pyreader():
            fetch_list = [self.loss.name]
            outputs = exe.run(fleet.main_program, feed=data, fetch_list=fetch_list)
            loss = outputs
            train_info.append(round(loss[0].tolist()[0], 6))
            steps += 1
            if steps >= 5:
                break
        print(train_info)
        return train_info


if __name__ == '__main__':
    runtime_main(TestDistBert)