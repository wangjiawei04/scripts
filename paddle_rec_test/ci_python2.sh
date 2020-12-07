#!/bin/bash
export LD_LIBRARY_PATH=/opt/rh/devtoolset-2/root/usr/lib64:/opt/rh/devtoolset-2/root/usr/lib:/usr/local/lib64:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
function setproxy(){
  export http_proxy=${proxy}
  export https_proxy=${proxy}
}

function check_style() {
  set -e

  export PATH=/usr/bin:$PATH
  pre-commit install

  if ! pre-commit run -a; then
    git diff
    exit 1
  fi

  exit 0
}

function prepare(){
    #setproxy
    cd PaddleRec
    pip uninstall paddle-rec -y
#     pip install skbuild 
#     pip install opencv-python==4.2.0.32 
    python setup.py install
    pip uninstall paddlepaddle -y
    wget https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl --no-check-certificate
#     wget https://paddle-wheel.bj.bcebos.com/2.0.0-rc0-gpu-cuda10.1-cudnn7-mkl_gcc8.2%2Fpaddlepaddle_gpu-2.0.0rc0.post101-cp27-cp27mu-linux_x86_64.whl --no-check-certificate
#     mv 2.0.0-rc0-gpu-cuda10.1-cudnn7-mkl_gcc8.2%2Fpaddlepaddle_gpu-2.0.0rc0.post101-cp27-cp27mu-linux_x86_64.whl paddlepaddle_gpu-2.0.0rc0.post101-cp27-cp27mu-linux_x86_64.whl
    
    #wget https://paddle-wheel.bj.bcebos.com/2.0.0-rc0-gpu-cuda10.1-cudnn7-mkl_gcc8.2%2Fpaddlepaddle_gpu-2.0.0rc0.post101-cp27-cp27mu-linux_x86_64.whl --no-check-certificate
#    wget https://paddle-wheel.bj.bcebos.com/1.8.5-gpu-cuda10-cudnn7-mkl%2Fpaddlepaddle_gpu-1.8.5.post107-cp27-cp27mu-linux_x86_64.whl --no-check-certificate
#    mv 1.8.5-gpu-cuda10-cudnn7-mkl%2Fpaddlepaddle_gpu-1.8.5.post107-cp27-cp27mu-linux_x86_64.whl paddlepaddle_gpu-1.8.5.post107-cp27-cp27mu-linux_x86_64.whl
    
#     wget https://paddle-wheel.bj.bcebos.com/1.7.2-gpu-cuda10-cudnn7-mkl%2Fpaddlepaddle_gpu-1.7.2.post107-cp27-cp27mu-linux_x86_64.whl --no-check-certificate
#     mv 1.7.2-gpu-cuda10-cudnn7-mkl%2Fpaddlepaddle_gpu-1.7.2.post107-cp27-cp27mu-linux_x86_64.whl paddlepaddle_gpu-1.7.2.post107-cp27-cp27mu-linux_x86_64.whl
#     wget https://paddle-wheel.bj.bcebos.com/1.8.3-gpu-cuda10-cudnn7-mkl%2Fpaddlepaddle_gpu-1.8.3.post107-cp27-cp27mu-linux_x86_64.whl --no-check-certificate
#     mv 1.8.3-gpu-cuda10-cudnn7-mkl%2Fpaddlepaddle_gpu-1.8.3.post107-cp27-cp27mu-linux_x86_64.whl paddlepaddle_gpu-1.8.3.post107-cp27-cp27mu-linux_x86_64.whl
    /opt/_internal/cpython-2.7.15-ucs4/bin/python -m pip install --upgrade pip
    /opt/_internal/cpython-2.7.15-ucs4/bin/python -m pip install paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
#    wget http://yq01-gpu-255-125-19-00.epc.baidu.com:8988/1.7.2/paddlepaddle_gpu-1.7.2.post107-cp27-cp27mu-linux_x86_64.whl
#    pip install paddlepaddle_gpu-1.7.2.post107-cp27-cp27mu-linux_x86_64.whl
#    wget http://yq01-gpu-255-125-19-00.epc.baidu.com:8988/paddlepaddle_gpu-0.0.0.2020.0803.102112.post107.develop-cp27-cp27mu-linux_x86_64.whl
#    pip install paddlepaddle_gpu-0.0.0.2020.0803.102112.post107.develop-cp27-cp27mu-linux_x86_64.whl
#     wget http://yq01-gpu-255-125-19-00.epc.baidu.com:8988/paddlepaddle_gpu-0.0.0.2020.0728.191705.post107.develop-cp27-cp27mu-linux_x86_64.whl
#     pip install paddlepaddle_gpu-0.0.0.2020.0728.191705.post107.develop-cp27-cp27mu-linux_x86_64.whl
#     wget https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
#     pip install paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
#    pip install paddlepaddle-gpu==1.8.2.post107
    pip install nose
    pip install ruamel.yaml
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
check_style
#run
