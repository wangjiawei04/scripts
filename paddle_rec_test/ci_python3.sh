#!/bin/bash

function setproxy(){
  export http_proxy=${proxy}
  export https_proxy=${proxy}
}

function prepare(){
    #setproxy
    cd PaddleRec
    python3 -m pip uninstall paddle-rec -y
    pip3 install skbuild 
    pip3 install opencv-python==4.2.0.32 
    python3 setup.py install
    python3 -m pip uninstall paddlepaddle -y
    wget https://paddle-wheel.bj.bcebos.com/1.8.5-gpu-cuda10-cudnn7-mkl%2Fpaddlepaddle_gpu-1.8.5.post107-cp35-cp35m-linux_x86_64.whl --no-check-certificate
    mv 1.8.5-gpu-cuda10-cudnn7-mkl%2Fpaddlepaddle_gpu-1.8.5.post107-cp35-cp35m-linux_x86_64.whl paddlepaddle_gpu-1.8.5.post107-cp35-cp35m-linux_x86_64.whl
    python3 -m pip install paddlepaddle_gpu-1.8.5.post107-cp35-cp35m-linux_x86_64.whl
#     wget https://paddle-wheel.bj.bcebos.com/1.8.3-gpu-cuda10-cudnn7-mkl%2Fpaddlepaddle_gpu-1.8.3.post107-cp35-cp35m-linux_x86_64.whl
#     mv 1.8.3-gpu-cuda10-cudnn7-mkl%2Fpaddlepaddle_gpu-1.8.3.post107-cp35-cp35m-linux_x86_64.whl paddlepaddle_gpu-1.8.3.post107-cp35-cp35m-linux_x86_64.whl
#     python3 -m pip install paddlepaddle_gpu-1.8.3.post107-cp35-cp35m-linux_x86_64.whl
#    wget http://yq01-gpu-255-125-19-00.epc.baidu.com:8988/1.7.2/paddlepaddle_gpu-1.7.2.post107-cp35-cp35m-linux_x86_64.whl
#    python3 -m pip install paddlepaddle_gpu-1.7.2.post107-cp35-cp35m-linux_x86_64.whl
#     wget https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-0.0.0-cp35-cp35m-linux_x86_64.whl
#     python3 -m pip install paddlepaddle_gpu-0.0.0-cp35-cp35m-linux_x86_64.whl
   # python3 -m pip install paddlepaddle-gpu==1.8.2.post107
    python3 -m pip install nose
    python3 -m pip install ruamel.yaml
    cd ../ && mkdir test_logs
    #   unset http_proxy, https_proxy
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
