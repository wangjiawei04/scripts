#!/bin/bash
function prepare(){
#    git clone http://github.com//PaddleRec.git
  #  pushd PaddleRec
    cd PaddleRec
    pip uninstall paddle-rec -y
    python setup.py install
    
    pip uninstall paddlepaddle -y
    wget https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
    pip install paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
   # pip install paddlepaddle-gpu==1.7.2.post107
    pip install nose
    pip install ruamel.yaml
 #   unset http_proxy, https_proxy
    cd ../ && mkdir test_logs
}

function run(){
    cases="test_paddlerec_features.py \
               test_paddlerec_features_new_config.py \
               test_paddlerec_mmoe.py \
               test_paddlerec_models.py \
               test_user_define.py"

    for file in ${cases}
    do
        echo ${file}
        #nosetests -s -v --with-html --html-report=test_logs/${file}.html ${file}
        nosetests -s -v ${file}
        rm -f *.yaml
        rm -rf increment* inference* logs
    done
}
prepare
#run
