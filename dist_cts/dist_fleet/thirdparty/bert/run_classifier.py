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
  * @file dist_bert.py
  * @author liyang109@baidu.com
  * @date 2020-05-13 15:53
  * @brief
  *
  **************************************************************************/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import subprocess
import multiprocessing

import paddle
import paddle.fluid as fluid

import reader.cls as reader
from model.bert import BertConfig
from model.classifier import create_model
from optimization import optimization
from utils.args import ArgumentGroup, print_arguments, check_cuda
from utils.init import init_pretraining_params, init_checkpoint
from utils.cards import get_cards
import dist_utils
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import paddle.fluid.incubate.fleet.base.role_maker as role_maker


from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from env import dist_env

num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
class main():
    def p_args(self):
        parser = argparse.ArgumentParser(__doc__)
        model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
        model_g.add_arg("bert_config_path",         str,  None,           "Path to the json file for bert model config.")
        model_g.add_arg("init_checkpoint",          str,  None,           "Init checkpoint to resume training from.")
        model_g.add_arg("init_pretraining_params",  str,  "uncased_L-24_H-1024_A-16/params",
                        "Init pre-training params which preforms fine-tuning from. If the "
                         "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")
        model_g.add_arg("checkpoints",              str,  "$PWD/tmp",  "Path to save checkpoints.")
        train_g = ArgumentGroup(parser, "training", "training options.")
        train_g.add_arg("epoch",             int,    3,       "Number of epoches for fine-tuning.")
        train_g.add_arg("learning_rate",     float,  5e-5,    "Learning rate used to train with warmup.")
        train_g.add_arg("lr_scheduler",      str,    "linear_warmup_decay",
                        "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
        train_g.add_arg("weight_decay",      float,  0.01,    "Weight decay rate for L2 regularizer.")
        train_g.add_arg("warmup_proportion", float,  0.1,
                        "Proportion of training steps to perform linear learning rate warmup for.")
        train_g.add_arg("loss_scaling",      float,  1.0,
                        "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")

        log_g = ArgumentGroup(parser,     "logging", "logging related.")
        log_g.add_arg("skip_steps",          int,    1,    "The steps interval to print loss.")
        log_g.add_arg("verbose",             bool,   False, "Whether to output verbose log.")

        data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
        data_g.add_arg("data_dir",      str,  "xnli",  "Path to training data.")
        data_g.add_arg("vocab_path",    str,  "uncased_L-24_H-1024_A-16/vocab.txt",  "Vocabulary path.")
        data_g.add_arg("max_seq_len",   int,  32,   "Number of words of the longest seqence.")
        data_g.add_arg("batch_size",    int,  5,    "Total examples' number in batch for training. see also --in_tokens.")
        data_g.add_arg("in_tokens",     bool, False,
                      "If set, the batch size will be the maximum number of tokens in one batch. "
                      "Otherwise, it will be the maximum number of examples in one batch.")
        data_g.add_arg("do_lower_case", bool, True,
                       "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
        data_g.add_arg("random_seed",   int,  0,     "Random seed.")
        data_g.add_arg("shuffle_seed",   int,  2,     "Shuffle seed.")

        run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
        run_type_g.add_arg("use_fast_executor",            bool,   False, "If set, use fast parallel executor (in experiment).")
        run_type_g.add_arg("shuffle",                      bool,   False,  "")
        run_type_g.add_arg("num_iteration_per_drop_scope", int,    1,     "Ihe iteration intervals to clean up temporary variables.")
        run_type_g.add_arg("task_name",                    str,    "XNLI",
                           "The name of task to perform fine-tuning, should be in {'xnli', 'mnli', 'cola', 'mrpc'}.")
        run_type_g.add_arg("do_train",                     bool,   True,  "Whether to perform training.")
        args = parser.parse_args()
        return args

    def net(self):
        args = self.p_args()
        bert_config = BertConfig("uncased_L-24_H-1024_A-16/bert_config.json")
        bert_config.print_config()
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = 1
        if args.do_train:
            my_dist_env = dist_env()
            worker_endpoints_env = my_dist_env["trainer_endpoints"]
            worker_endpoints = worker_endpoints_env.split(",")
            current_endpoint = my_dist_env["current_endpoint"]
            trainer_id = worker_endpoints.index(current_endpoint)
            # new rolemaker here
            print("current_id: ", trainer_id)
            print("worker_endpoints: ", worker_endpoints)
            role = role_maker.UserDefinedCollectiveRoleMaker(
            current_id=trainer_id,
            worker_endpoints=worker_endpoints)
            # Fleet get role of each worker
            fleet.init(role)
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

        dev_count = len(worker_endpoints)
        # we need to keep every trainer of fleet the same shuffle_seed
        print("shuffle_seed: ", args.shuffle_seed)
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
        self.warmup_steps = int(5 * 0.1)

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.use_experimental_executor = args.use_fast_executor
        exec_strategy.num_threads = dev_count
        exec_strategy.num_iteration_per_drop_scope = args.num_iteration_per_drop_scope

        dist_strategy = DistributedStrategy()
        dist_strategy.exec_strategy = exec_strategy
        dist_strategy.nccl_comm_num = 3
        dist_strategy.use_hierarchical_allreduce = True
        #dist_strategy.mode = "collective"
        #dist_strategy.collective_mode = "grad_allreduce"

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
                    dist_strategy = dist_strategy)

        exe.run(startup_prog)
        with open("__model__", "wb") as f:
            f.write(fleet._origin_program.desc.serialize_to_string())

        with open("debug_program", "w") as f:
            f.write(str(fleet._origin_program))
        return self.loss

    def do_training(self):
        args = self.p_args()
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        exe = fluid.Executor(place)
        self.train_pyreader.decorate_batch_generator(self.train_data_generator, place)
        steps = 0
        total_cost, total_acc, total_num_seqs = [], [], []
        time_begin = time.time()
        throughput = []
        ce_info = []
        train_info=[]
        if True:
            for data in self.train_pyreader():
                if steps % 1 == 0:
                    if self.warmup_steps <= 0:
                        fetch_list = [self.loss.name]
                    else:
                        fetch_list = [self.loss.name]
                else:
                    fetch_list = []
                outputs = exe.run(fleet.main_program, feed=data, fetch_list=fetch_list)
                loss = outputs
                print(type(loss))
               # train_info.append(loss)
                # print(loss)
                # if steps % 10 == 0:
                #     if self.warmup_steps <= 0:
                #         loss = outputs
                #     else:
                #         loss = outputs
                time_end = time.time()
                used_time = time_end - time_begin
                steps += 1
                if steps > 10:
                    break
        return train_info


if __name__ == '__main__':
    a = main()
    a.net()
    a.do_training()

