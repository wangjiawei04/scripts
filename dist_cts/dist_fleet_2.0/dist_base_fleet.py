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
  * @file dist_base_fleet.py
  * @author liyang109@baidu.com
  * @date 2020-11-05 15:53
  * @brief
  *
  **************************************************************************/
"""
import argparse
import os
import pickle
import json
import signal
import subprocess
import sys
import time
import traceback
import paddle
import decorator
import paddle.fluid as fluid
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker


RUN_STEP = 5
LEARNING_RATE = 0.01
paddle.enable_static()


class FleetDistRunnerBase(object):
    """dist fleet case runner base."""

    def __init__(self, batch_num=5, batch_size=32):
        self.batch_num = batch_num
        self.batch_size = batch_size

    def _set_strategy(self, args):
        """配置运行的distributed_strategy,
           build_strategy 配置在do_training中"""
        self.dist_strategy = fleet.DistributedStrategy()
        if args.run_params["sync_mode"] == "sync":
            self.dist_strategy.a_sync = False
        elif args.run_params["sync_mode"] == "async":
            self.dist_strategy.a_sync = True
        elif args.run_params["sync_mode"] == "geo_async":
            self.dist_strategy.a_sync = True
            self.dist_strategy.a_sync_configs = {
                "k_steps": 2
            }
        elif args.run_params["sync_mode"] == "auto":
            self.dist_strategy.auto = True


    def run_pserver(self, args):
        """
        run pserver process, you don't need to implement it.
        Args:
            args (ArgumentParser): run args to config dist fleet.
        """
        import paddle.distributed.fleet as fleet
        if args.role.upper() != "PSERVER":
            raise ValueError("args role must be PSERVER")
        fleet.init()
        self._set_strategy(args)
        avg_cost = self.net(args)
        optimizer = fluid.optimizer.SGD(LEARNING_RATE)
        optimizer = fleet.distributed_optimizer(optimizer, self.dist_strategy)
        optimizer.minimize(avg_cost)
        fleet.init_server()
        fleet.run_server()

    def run_trainer(self, args):
        """
        run trainer process, you don't need to implement it.
        Args:
            args (ArgumentParser): run args to config dist fleet.
        """
        import paddle.distributed.fleet as fleet
        if args.role.upper() != "TRAINER":
            raise ValueError("args role must be TRAINER")
        fleet.init()
        self._set_strategy(args)
        avg_cost = self.net(args)
        optimizer = fluid.optimizer.SGD(LEARNING_RATE)
        optimizer = fleet.distributed_optimizer(optimizer, self.dist_strategy)
        optimizer.minimize(avg_cost)
        # if args.run_params.get("run_from_dataset", False):
        losses = self.do_training_from_dataset(fleet, args)
        # else:
        #     losses = self.do_training(fleet, args)
        losses = "" if not losses else losses
        print(losses)

    def run_nccl_trainer(self, args):
        """
        run nccl trainer, used for gpu case.
        Args:
            args (ArgumentParser): run args to config dist fleet.
        """
        assert args.update_method == "nccl"
        import paddle.distributed.fleet as fleet
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = args.run_params['num_threads']
        if args.role.upper() != "TRAINER":
            raise ValueError("args role must be TRAINER")
        fleet.init(is_collective=True)
        avg_cost = self.net(args)
        losses = self.do_training(fleet, args)
        losses = "" if not losses else losses
        print(losses)

    def net(self, args=None):
        """
        construct model's net. Each model has its own unique network.
        Args:
            args (ArgumentParser): run args to config dist fleet.
        """
        raise NotImplementedError(
            "get_model should be implemented by child classes.")

    def do_training(self, fleet, args=None):
        """
        training from pyreader.
        Args:
            fleet:
            args (ArgumentParser): run args to config dist fleet.
        """
        raise NotImplementedError(
            "do_training should be implemented by child classes.")

    def do_training_from_dataset(self, fleet, args=None):
        """
        training from dataset.
        Args:
            fleet:
            args (ArgumentParser): run args to config dist fleet.
        """
        raise NotImplementedError(
            "do_training should be implemented by child classes.")

    def py_reader(self):
        """use py_reader."""
        raise NotImplementedError(
            "py_reader should be implemented by child classes.")

    def dataset_reader(self):
        """use dataset_reader."""
        raise NotImplementedError(
            "dataset_reader should be implemented by child classes.")


class TestFleetBase(object):
    """TestDistRun."""

    def __init__(self, pservers=2, trainers=2):
        self.trainers = trainers
        self.pservers = pservers
        self.ps_endpoints = ""
        for i in range(pservers):
            self.ps_endpoints += "127.0.0.1:912%s," % (i + 1)
        self.ps_endpoints = self.ps_endpoints[:-1]
        self.python_interp = "python"
        self.run_params = {}

    def start_pserver(self, model_file, check_error_log):
        """
        start_pserver
        Args:
            model_file (str):
            check_error_log (bool):
        Returns:
            ([], [])
        """
        ps_endpoint_list = self.ps_endpoints.split(",")
        ps_pipe_list = []
        ps_proc_list = []
        run_params = json.dumps(self.run_params).replace(" ", "")

        for i, _ in enumerate(ps_endpoint_list):
            ps_cmd = "{} {} --update_method pserver --role pserver --endpoints {} " \
                     "--current_id {} --trainers {} --run_params {}".format(
                     self.python_interp,
                     model_file,
                     self.ps_endpoints,
                     i,
                     self.trainers,
                     run_params)
            ps_pipe = subprocess.PIPE
            if check_error_log:
                # print("ps_cmd:", ps_cmd)
                ps_pipe = open(
                    os.path.join(
                        os.getenv("LOG_PATH", '/tmp'), "ps%s_err.log" % i),
                    "wb")
            ps_proc = subprocess.Popen(
                ps_cmd.split(" "), stdout=subprocess.PIPE, stderr=ps_pipe)
            ps_pipe_list.append(ps_pipe)
            ps_proc_list.append(ps_proc)
        return ps_proc_list, ps_pipe_list

    def _wait_ps_ready(self, pid):
        retry_times = 5
        return
        # while True:
        #     assert retry_times >= 0, "wait ps ready failed"
        #     time.sleep(3)
        #     try:
        #         # the listen_and_serv_op would touch a file which contains the listen port
        #         # on the /tmp directory until it was ready to process all the RPC call.
        #         os.stat(os.path.join('/tmp', "paddle.%d.port" % pid))
        #         return
        #     except os.error as e:
        #         sys.stderr.write('waiting for pserver: %s, left retry %d\n' %
        #                          (e, retry_times))
        #         retry_times -= 1

    def get_result(self,
                   model_file,
                   check_error_log=True,
                   update_method="pserver",
                   gpu_num=0,
                   models_change_env=None):
        """
        get result.
        Args:
            model_file (str):
            check_error_log (bool):
            update_method (str):
            gpu_num (int):
            models_change_env (dict):
        Returns:
            list
        """
        run_params = json.dumps(self.run_params).replace(" ", "")
        flags_params = json.loads(run_params)
        required_envs = {
            "PATH": os.getenv("PATH"),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ''),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "FLAGS_fraction_of_gpu_memory_to_use": "0.15",
            "FLAGS_cudnn_deterministic": "1",
            "FLAGS_rpc_deadline": "5000",  # 5sec to fail fast
            "http_proxy": "",
        }
        if "run_from_dataset" in flags_params:
            required_envs["CPU_NUM"] = str(flags_params['cpu_num'])

        if check_error_log:
            required_envs["GLOG_v"] = "1"
            required_envs["GLOG_logtostderr"] = "1"

        if models_change_env:
            required_envs.update(models_change_env)
        # Run local to get a base line
        if update_method == "pserver":
            ps_proc_list, _ = self.start_pserver(model_file, check_error_log)
            for ps_proc in ps_proc_list:
                self._wait_ps_ready(ps_proc.pid)
        tr_cmd_lists = []
        tr_proc_list = []
        FNULL = open(os.devnull, 'w')
        if update_method == "pserver":
            for i in range(self.trainers):
                tr_cmd = "{} {} --update_method pserver --role trainer --endpoints {} " \
                         "--current_id {} --trainers {} --run_params {}".format(
                    self.python_interp,
                    model_file,
                    self.ps_endpoints,
                    i,
                    self.trainers,
                    run_params)
                devices_define = ""
                for j in range(gpu_num):
                    devices_define += "%d," % (i * gpu_num + j)

                envs = {}
                envs.update(required_envs)
                tr_cmd_lists.append({"cmd": tr_cmd, "envs": envs})
        elif update_method == "nccl":
            all_nodes_devices_endpoints = ""
            nranks = self.trainers * gpu_num
            for trainer_id in range(self.trainers):
                for i in range(gpu_num):
                    if all_nodes_devices_endpoints:
                        all_nodes_devices_endpoints += ","
                    all_nodes_devices_endpoints += "127.0.0.1:617%d" % (
                        trainer_id * gpu_num + i)
            for real_id in range(nranks):
                envs = {}
                envs.update(required_envs)
                envs.update({
                    "PADDLE_TRAINER_ID": "%d" % real_id,
                    "PADDLE_CURRENT_ENDPOINT":
                    "%s:617%d" % ("127.0.0.1", real_id),
                    "PADDLE_TRAINERS_NUM": "%d" % nranks,
                    "PADDLE_TRAINER_ENDPOINTS": all_nodes_devices_endpoints,
                    "FLAGS_selected_gpus": "%d" % real_id,
                    "FLAGS_fraction_of_gpu_memory_to_use": "0.96",
                    "FLAGS_eager_delete_tensor_gb": "0",
                    "FLAGS_fuse_parameter_memory_size": "16",
                    "FLAGS_fuse_parameter_groups_size": "50",
                    "FLAGS_cudnn_deterministic": "0",
                })
                tr_cmd = "{} {} --update_method nccl --role trainer --endpoints {} " \
                         "--current_id {} --trainers {} --run_params {}".format(
                          self.python_interp,
                          model_file,
                          all_nodes_devices_endpoints,
                          real_id,
                          nranks,
                          run_params)
                nccl_devs = {
                    "NCCL_SOCKET_IFNAME": "eth0",
                    "NCCL_P2P_DISABLE": "1",
                    "NCCL_IB_DISABLE": "0",
                    "NCCL_IB_CUDA_SUPPORT": "1"
                }
                envs.update(nccl_devs)
                tr_cmd_lists.append({"cmd": tr_cmd, "envs": envs})
        for real_id, tr_cmd_dict in enumerate(tr_cmd_lists):
            tr_cmd = tr_cmd_dict["cmd"]
            envs = tr_cmd_dict["envs"]
            # print tr_cmd, envs
            tr_pipe = subprocess.PIPE
            if check_error_log:
                #    print("tr_cmd:", tr_cmd)
                tr_pipe = open(
                    os.path.join(
                        os.getenv("LOG_PATH", '/tmp'),
                        "tr%s_err.log" % real_id), "wb")
            tr_proc = subprocess.Popen(
                tr_cmd.split(" "),
                stdout=subprocess.PIPE,
                stderr=tr_pipe,
                env=envs)
            tr_proc_list.append(tr_proc)
        for tr_proc in tr_proc_list:
            tr_proc.wait()
        # train data
        train_data = []
        try:
            for tr_proc in tr_proc_list:
                out, _ = tr_proc.communicate()
                lines = out.split(b"\n")[-2]
                if lines:
                    lines = lines[1:-2].split(b",")
                loss = [eval(i) for i in lines]
                train_data.append(loss)
        except Exception:
            traceback.print_exc()
            train_data = []
        finally:
            if update_method == "pserver":
                for ps_proc in ps_proc_list:
                    os.kill(ps_proc.pid, signal.SIGKILL)
            FNULL.close()
        print(train_data)
        return train_data


def runtime_main(test_class):
    """
    run main test_class
    Args:
        test_class (FleetDistRunnerBase):
    """
    parser = argparse.ArgumentParser(description='Run Fleet test.')
    parser.add_argument(
        '--update_method', type=str, required=True,
        choices=['pserver', 'nccl'])
    parser.add_argument(
        '--role', type=str, required=True, choices=['pserver', 'trainer'])
    parser.add_argument('--endpoints', type=str, required=False, default="")
    parser.add_argument('--current_id', type=int, required=False, default=0)
    parser.add_argument('--trainers', type=int, required=False, default=1)
    parser.add_argument('--run_params', type=str, required=True, default='{}')
    args = parser.parse_args()
    args.run_params = json.loads(args.run_params)
    model = test_class()
    if args.update_method == "nccl":
        model.run_nccl_trainer(args)
        return
    if args.role == "pserver":
        model.run_pserver(args)
    else:
        model.run_trainer(args)


def run_by_freq(freq):
    """testcase run by frequency, it contains DAILY, MONTH."""

    @decorator.decorator
    def wrapper(func, *args, **kwargs):
        """run daily or month"""
        if os.getenv("RUN_FREQUENCY", "DAILY") == freq:
            return func(*args, **kwargs)
        else:
            return

    return wrapper


@decorator.decorator
def run_with_compatibility(func, *args, **kwargs):
    """test case run with compatibility paddle version."""
    os.environ["PADDLE_COMPATIBILITY_CHECK"] = "1"
    func(*args, **kwargs)
    os.environ["PADDLE_COMPATIBILITY_CHECK"] = "0"