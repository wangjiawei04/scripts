#!/bin/bash

echo "################################################################"
echo "#                                                              #"
echo "#                                                              #"
echo "#                                                              #"
echo "#          Paddle Serving  begin run!!!                        #"
echo "#                                                              #"
echo "#                                                              #"
echo "#                                                              #"
echo "################################################################"

build_path=/workspace/Serving/Serving
build_whl_list=(build_gpu_server build_client build_cpu_server build_app)
rpc_model_list=(bert_rpc_gpu bert_rpc_cpu faster_rcnn_model_rpc ResNet50_rpc ResNet101_rpc lac_rpc cnn_rpc bow_rpc lstm_rpc fit_a_line_rpc)

function setproxy(){
  export http_proxy=${proxy}
  export https_proxy=${proxy}
}

function unsetproxy(){
  unset http_proxy
  unset https_proxy
}

function kill_server_process(){
  kill `ps -ef|grep serving|awk '{print $2}'`
}

function check() {
    cd ${build_path}
    if [ ! -f paddle_serving_app* ]; then
      echo "paddle_serving_app is compiled failed, please check your pull request"
      exit 1
    elif [ ! -f paddle_serving_server-* ]; then
      echo "paddle_serving_server-cpu is compiled failed, please check your pull request"
      exit 1
    elif [ ! -f paddle_serving_server_* ]; then
      echo "paddle_serving_server_gpu is compiled failed, please check your pull request"
      exit 1
    elif [ ! -f paddle_serving_client* ]; then
      echo "paddle_serving_server_client is compiled failed, please check your pull request"
      exit 1
    else
      echo "paddle serving build passed"
    fi
}

function before_hook(){
  setproxy
  cd /workspace/Serving/Serving
  pip3 install --upgrade pip
  pip3 install numpy==1.16.4 sentencepiece
  echo "env configuration succ.... "
}

function run_env(){
  setproxy
  yum install -y libXext libSM libXrender
  pip3 install --upgrade nltk==3.4
  pip3 install --upgrade scipy==1.2.1
  pip3 install --upgrade setuptools==41.0.0
  pip3 install paddlehub ujson paddlepaddle
  echo "env configuration succ.... "
}

function run_gpu_env(){
  cd ${build_path}
  if [ -d build ];then
    rm -rf build
  fi
  cp -r ${build_path}/build_gpu/ ${build_path}/build
}

function run_cpu_env(){
  cd ${build_path}
  if [ -d build ];then
    rm -rf build
  fi
  cp -r ${build_path}/build_cpu/ ${build_path}/build
}

function build_gpu_server() {
    setproxy
    cd ${build_path}
    git submodule update --init --recursive
    if [ -d build ];then
        cd build && rm -rf *
    else
      mkdir build && cd build
    fi
    cmake -DPYTHON_INCLUDE_DIR=$PYTHONROOT/include/python3.6m/ \
          -DPYTHON_LIBRARIES=$PYTHONROOT/lib64/libpython3.6.so \
          -DPYTHON_EXECUTABLE=$PYTHONROOT/bin/python3 \
          -DSERVER=ON \
          -DWITH_GPU=ON ..
    make -j10
    make -j10
    make install -j10
    pip3 install ${build_path}/build/python/dist/*
    cp  ${build_path}/build/python/dist/* ../
    cp -r ${build_path}/build/ ${build_path}/build_gpu
}

function build_client() {
     setproxy
     cd  ${build_path}
     if [ -d build ];then
          cd build && rm -rf *
      else
        mkdir build && cd build
      fi
     cmake -DPYTHON_INCLUDE_DIR=$PYTHONROOT/include/python3.6m/ \
           -DPYTHON_LIBRARIES=$PYTHONROOT/lib64/libpython3.6.so \
           -DPYTHON_EXECUTABLE=$PYTHONROOT/bin/python3 \
           -DCLIENT=ON ..
     make -j10
     make -j10
     cp ${build_path}/build/python/dist/* ../
     pip3 install ${build_path}/build/python/dist/*
}

function build_cpu_server(){
      setproxy
      cd ${build_path}
      if [ -d build ];then
          cd build && rm -rf *
      else
        mkdir build && cd build
      fi
      cmake -DPYTHON_INCLUDE_DIR=$PYTHONROOT/include/python3.6m/ \
            -DPYTHON_LIBRARIES=$PYTHONROOT/lib64/libpython3.6.so \
            -DPYTHON_EXECUTABLE=$PYTHONROOT/bin/python3 \
            -DSERVER=ON ..
      make -j10
      make -j10
      make install -j10
      cp ${build_path}/build/python/dist/* ../
      pip3 install ${build_path}/build/python/dist/*
      cp -r ${build_path}/build/ ${build_path}/build_cpu
}

function build_app() {
  setproxy
  pip3 install paddlepaddle paddlehub ujson paddle_serving_client opencv-python Pillow
  yum install -y libXext libSM libXrender
  cd ${build_path}
  if [ -d build ];then
      cd build && rm -rf *
  else
    mkdir build && cd build
  fi
  cmake -DPYTHON_INCLUDE_DIR=$PYTHONROOT/include/python3.6m/ \
        -DPYTHON_LIBRARIES=$PYTHONROOT/lib/libpython3.6.so \
        -DPYTHON_EXECUTABLE=$PYTHONROOT/bin/python3 \
        -DCMAKE_INSTALL_PREFIX=./output -DAPP=ON ..
  make
  cp ${build_path}/build/python/dist/* ../
  pip3 install ${build_path}/build/python/dist/*
}

function bert_rpc_gpu(){
  run_gpu_env
  setproxy
  cd ${build_path}/python/examples/bert
  sh get_data.sh >/dev/null 2>&1
  sed -i "34cendpoint_list = ['${host}:8860']" bert_client.py
  sed -i '$aprint(result)' bert_client.py
  python3 prepare_model.py 128
  sleep 3
  python3 -m paddle_serving_server_gpu.serve --model bert_seq128_model/ --port 8860 --gpu_ids 0 > bert_rpc_gpu 2>&1 &
  sleep 10
  head data-c.txt | python3 bert_client.py --model bert_seq128_client/serving_client_conf.prototxt
  kill_server_process
}

function bert_rpc_cpu(){
  run_cpu_env
  setproxy
  cd ${build_path}/python/examples/bert
  sed -i "34cendpoint_list = ['${host}:8861']" bert_client.py
  python3 -m paddle_serving_server.serve --model bert_seq128_model/ --port 8861 > bert_rpc_cpu 2>&1 &
  sleep 3
  cp data-c.txt.1 data-c.txt
  head data-c.txt | python3 bert_client.py --model bert_seq128_client/serving_client_conf.prototxt
  kill_server_process
}

function criteo_ctr_rpc(){
  setproxy
  run_cpu_env
  cd ${build_path}/python/examples/criteo_ctr
  sh get_data.sh >/dev/null 2>&1
  wget https://paddle-serving.bj.bcebos.com/criteo_ctr_example/criteo_ctr_demo_model.tar.gz >/dev/null 2>&1
  tar xf criteo_ctr_demo_model.tar.gz >/dev/null 2>&1
  mv models/ctr_client_conf .
  mv models/ctr_serving_model .
  sed -i "30cclient.connect(['${host}:8862'])" test_client.py
  python3 -m paddle_serving_server_gpu.serve --model ctr_serving_model/ --port 8862 --gpu_ids 0 > criteo_ctr_rpc 2>&1 &
  sleep 5
  python3 test_client.py ctr_client_conf/serving_client_conf.prototxt raw_data/ >ctr_log 2>&1
  tailf ctr_log
  kill_server_process
  sleep 5
}

function criteo_ctr_gpu_rpc(){
  run_gpu_env
  setproxy
  cd ${build_path}/python/examples/criteo_ctr
  python3 -m paddle_serving_server_gpu.serve --model ctr_serving_model/ --port 8862 --gpu_ids 0 > criteo_ctr_rpc_gpu 2>&1 &
  sleep 3
  python3 test_client.py ctr_client_conf/serving_client_conf.prototxt raw_data/ > ctr_gpu_log 2>&1
  kill_server_process
  sleep 3
}

function ResNet50_rpc(){
  run_gpu_env
  setproxy
  cd ${build_path}/python/examples/imagenet
  sh get_model.sh >/dev/null 2>&1
  sed -i "23cclient.connect(['127.0.0.1:8863'])" resnet50_rpc_client.py
  python3 -m paddle_serving_server_gpu.serve --model ResNet50_vd_model --port 8863 --gpu_ids 0 > ResNet50_rpc 2>&1 &
  sleep 5
  python3 resnet50_rpc_client.py ResNet50_vd_client_config/serving_client_conf.prototxt
  kill_server_process
  sleep 5
}

function ResNet101_rpc(){
  run_gpu_env
  setproxy
  cd ${build_path}/python/examples/imagenet
  sed -i "22cclient.connect(['${host}:8864'])" image_rpc_client.py
  python3 -m paddle_serving_server_gpu.serve --model ResNet101_vd_model --port 8864 --gpu_ids 0 > ResNet101_rpc 2>&1 &
  sleep 5
  python3 image_rpc_client.py ResNet101_vd_client_config/serving_client_conf.prototxt
  kill_server_process
  sleep 5
}

function cnn_rpc(){
  setproxy
  run_cpu_env
  cd ${build_path}/python/examples/imdb
  sh get_data.sh >/dev/null 2>&1
  sed -i "21cclient.connect(['${host}:8865'])" test_client.py
  python3 -m paddle_serving_server.serve --model imdb_cnn_model/ --port 8865 > cnn_rpc 2>&1 &
  sleep 5
  head test_data/part-0 | python3 test_client.py imdb_cnn_client_conf/serving_client_conf.prototxt imdb.vocab
  kill_server_process
}

function bow_rpc(){
  setproxy
  run_cpu_env
  cd ${build_path}/python/examples/imdb
  sed -i "21cclient.connect(['${host}:8866'])" test_client.py
  python3 -m paddle_serving_server.serve --model imdb_bow_model/ --port 8866 > bow_rpc 2>&1 &
  sleep 5
  head test_data/part-0 | python3 test_client.py imdb_bow_client_conf/serving_client_conf.prototxt imdb.vocab
  kill_server_process
  sleep 5
}

function lstm_rpc(){
  setproxy
  run_cpu_env
  cd ${build_path}/python/examples/imdb
  sed -i "21cclient.connect(['${host}:8867'])" test_client.py
  python3 -m paddle_serving_server.serve --model imdb_lstm_model/ --port 8867 > lstm_rpc 2>&1 &
  sleep 5
  head test_data/part-0 | python3 test_client.py imdb_lstm_client_conf/serving_client_conf.prototxt imdb.vocab
  kill_server_process
  sleep 5
}

function lac_rpc(){
  setproxy
  run_cpu_env
  cd ${build_path}/python/examples/lac
  sh get_data.sh >/dev/null 2>&1
  sed -i "25cclient.connect(['${host}:8868'])" lac_client.py
  python3 -m paddle_serving_server.serve --model jieba_server_model/ --port 8868 > lac_rpc 2>&1 &
  sleep 5
  echo "我爱北京天安门" | python3 lac_client.py jieba_client_conf/serving_client_conf.prototxt lac_dict/
  kill_server_process
  sleep 5
}

function fit_a_line_rpc(){
  setproxy
  run_cpu_env
  cd ${build_path}/python/examples/fit_a_line
  sh get_data.sh >/dev/null 2>&1
  sed -i "35cserver.prepare_server(workdir='work_dir1', port=8869, device='cpu')" test_server.py
  sed -i "21cclient.connect(['${host}:8869'])" test_client.py
  python3 test_server.py uci_housing_model/ > line_rpc 2>&1 &
  sleep 5
  python3 test_client.py uci_housing_client/serving_client_conf.prototxt
  kill_server_process
}

function faster_rcnn_model_rpc(){
  run_gpu_env
  setproxy
  kill_server_process
  cd ${build_path}/python/examples/faster_rcnn_model
  wget https://paddle-serving.bj.bcebos.com/pddet_demo/faster_rcnn_model.tar.gz >/dev/null 2>&1
  wget https://paddle-serving.bj.bcebos.com/pddet_demo/infer_cfg.yml >/dev/null 2>&1
  tar xf faster_rcnn_model.tar.gz >/dev/null 2>&1
  mv faster_rcnn_model/pddet* ./
  sed -i "30s/127.0.0.1:9393/${host}:8870/g" new_test_client.py
  python3 -m paddle_serving_server_gpu.serve --model pddet_serving_model --port 8870 --gpu_id 0 > haha 2>&1 &
  sleep 3
  python3 new_test_client.py pddet_client_conf/serving_client_conf.prototxt infer_cfg.yml 000000570688.jpg
  kill_server_process
  sleep 2
}


function build_all_whl(){
  for whl in ${build_whl_list[@]}
  do
    echo "===========${whl} begin build==========="
    $whl
    sleep 3
    echo "===========${whl} build over ==========="
  done
}

function run_rpc_models(){
  for model in ${rpc_model_list[@]}
  do
    echo "===========${model} run begin==========="
    $model
    sleep 3
    echo "===========${model} run  end ==========="
  done
}

function end_hook(){
  cd ${build_path}
  echo "===========files==========="
  ls -hlst
  echo "=========== end ==========="
}

function main() {
  before_hook
  build_all_whl
  check
  run_env
  run_rpc_models
  #criteo_ctr_rpc 
  #criteo_ctr_gpu_rpc
  end_hook
}


main$@
