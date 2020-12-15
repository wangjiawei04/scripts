```bash
paddlerec测试case复现

1：git clone https://github.com/gentelyang/scripts.git
2：cd scripts
3：git clone https://github.com/PaddlePaddle/PaddleRec.git
4: cd PaddleRec
5：git fetch origin pull/pr号/head:test  
6：git checkout test
7：pip uninstall paddlepaddle paddlepaddle-gpu -y
8: python setup install
9: pip uninstall paddlepaddle paddlepaddle-gpu -y
10: if PY2:
       wget https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl --no-check-certificate
       pip install paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
    else:
       wget https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl --no-check-certificate
       pip3 install paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
 
11：python -m pip install nose
12：python -m pip install ruamel.yaml
13：nosetests -s -v test_paddlerec_features_new_config.yaml
14: nosetests -s -v ...
```
