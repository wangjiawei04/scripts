#!/bin/bash
cur_time=`date +"%Y%m%d%H%M"`
job_name=bert${cur_time}
# 作业参数
group_name="dltp-0-yq01-k8s-gpu-v100-8" # 将作业提交到group_name指定的组，必填
job_version="paddle-fluid-v1.7.1"
start_cmd="sh bert_large_max_batch_size.sh"
k8s_gpu_cards=8
wall_time="10:00:00" #最大运行时间
k8s_priority="high"
file_dir="."
paddlecloud job train --job-name ${job_name} \
   --job-conf config.ini \
   --group-name ${group_name} \
   --start-cmd "${start_cmd}" \
   --file-dir ${file_dir} \ #使用 —file-dir提交任务(保留file-dir的完整目录结构)
   --job-version ${job_version} \
   --k8s-gpu-cards ${k8s_gpu_cards} \
   --k8s-priority ${k8s_priority} \
   --wall-time ${wall_time} \
   --is-standalone 1
