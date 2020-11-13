#!/bin/bash

echo "################################################################"
echo "#                                                              #"
echo "#                                                              #"
echo "#                                                              #"
echo "#          Paddle Serving  begin run  with python3.6.8!!!      #"
echo "#                                                              #"
echo "#                                                              #"
echo "#                                                              #"
echo "################################################################"

build_path=/workspace/Serving/
build_whl_list=(build_gpu_server build_client build_cpu_server build_app)
rpc_model_list=(bert_rpc_gpu bert_rpc_cpu faster_rcnn_model_rpc ResNet50_rpc lac_rpc cnn_rpc bow_rpc lstm_rpc \ 
fit_a_line_rpc cascade_rcnn_rpc deeplabv3_rpc mobilenet_rpc unet_rpc resnetv2_rpc ocr_rpc)
http_model_list=(fit_a_line_http lac_http cnn_http bow_http lstm_http ResNet50_http bert_http)


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

function check_result() {
    if [ $? -ne 0 ];then
       echo "$1 model runs failed, please check your pull request or modify test case!"
       exit 1
    else
       echo "$1 model runs successfully, congratulations!"
    fi
}

function before_hook(){
  setproxy
  cd /workspace/Serving/
  pip3 install --upgrade pip
  pip3 install numpy==1.16.4 sentencepiece
  pip3 install grpcio-tools==1.28.1
  pip3 install grpcio==1.28.1
  echo "env configuration succ.... "
}

function run_env(){
  setproxy
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
  sed -i "26cendpoint_list = ['${host}:8860']" bert_client.py
  sed -i '$aprint(result)' bert_client.py
  wget https://sys-p0.bj.bcebos.com/bert_seq128_model.tar.gz >/dev/null 2>&1
  wget https://sys-p0.bj.bcebos.com/bert_seq128_client.tar.gz >/dev/null 2>&1
  tar -zxvf bert_seq128_model.tar.gz >/dev/null 2>&1
  tar -zxvf bert_seq128_client.tar.gz >/dev/null 2>&1
  sleep 3
  ls -hlst
  python3 -m paddle_serving_server_gpu.serve --model bert_seq128_model/ --port 8860 --gpu_ids 0 > bert_rpc_gpu 2>&1 &
  sleep 15
  tail bert_rpc_gpu
  head data-c.txt | python3 bert_client.py --model bert_seq128_client/serving_client_conf.prototxt
  check_result $FUNCNAME
  kill_server_process
}

function bert_rpc_cpu(){
  run_cpu_env
  setproxy
  cd ${build_path}/python/examples/bert
  sed -i "26cendpoint_list = ['${host}:8861']" bert_client.py
  python3 -m paddle_serving_server.serve --model bert_seq128_model/ --port 8861 > bert_rpc_cpu 2>&1 &
  sleep 3
  cp data-c.txt.1 data-c.txt
  head data-c.txt | python3 bert_client.py --model bert_seq128_client/serving_client_conf.prototxt
  check_result $FUNCNAME
  kill_server_process
}

# function bert_http() {

# }
function criteo_ctr_rpc(){
  setproxy
  run_cpu_env
  cd ${build_path}/python/examples/criteo_ctr
  wget --no-check-certificate https://sys-p0.bj.bcebos.com/models/ctr_data.tar.gz >/dev/null 2>&1
  tar -zxf ctr_data.tar.gz
  wget https://paddle-serving.bj.bcebos.com/criteo_ctr_example/criteo_ctr_demo_model.tar.gz >/dev/null 2>&1
  tar xf criteo_ctr_demo_model.tar.gz >/dev/null 2>&1
  mv models/ctr_client_conf .
  mv models/ctr_serving_model .
  sed -i "30cclient.connect(['${host}:8862'])" test_client.py
  python3 -m paddle_serving_server_gpu.serve --model ctr_serving_model/ --port 8862 --gpu_ids 0 > criteo_ctr_rpc 2>&1 &
  sleep 5
  python3 test_client.py ctr_client_conf/serving_client_conf.prototxt raw_data/ >ctr_log 2>&1
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
  check_result $FUNCNAME
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
  #sh get_data.sh >/dev/null 2>&1
  python3 -m paddle_serving_app.package --get_model lac >/dev/null 2>&1
  tar -xzvf lac.tar.gz >/dev/null 2>&1
  sed -i "25cclient.connect(['${host}:8868'])" lac_client.py
  python3 -m paddle_serving_server.serve --model lac_model/ --port 8868 > lac_rpc 2>&1 &
  sleep 5
  echo "我爱北京天安门" | python3 lac_client.py lac_client/serving_client_conf.prototxt lac_dict/
  check_result $FUNCNAME
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
  check_result $FUNCNAME
  kill_server_process
}

function faster_rcnn_model_rpc(){
  setproxy
  run_gpu_env
  cd ${build_path}/python/examples/faster_rcnn_model
  wget https://paddle-serving.bj.bcebos.com/pddet_demo/faster_rcnn_model.tar.gz >/dev/null 2>&1
  wget https://paddle-serving.bj.bcebos.com/pddet_demo/infer_cfg.yml >/dev/null 2>&1
  tar xf faster_rcnn_model.tar.gz >/dev/null 2>&1
  mv faster_rcnn_model/pddet* ./
  sed -i "30s/127.0.0.1:9494/${host}:8870/g" test_client.py
  python3 -m paddle_serving_server_gpu.serve --model pddet_serving_model --port 8870 --gpu_id 0 > haha 2>&1 &
  echo "faster rcnn running ..."
  sleep 5
  python3 test_client.py pddet_client_conf/serving_client_conf.prototxt infer_cfg.yml 000000570688.jpg
  check_result $FUNCNAME
  kill_server_process
}

function cascade_rcnn_rpc(){
  setproxy
  run_gpu_env
  cd ${build_path}/python/examples/cascade_rcnn
  wget --no-check-certificate https://paddle-serving.bj.bcebos.com/pddet_demo/cascade_rcnn_r50_fpx_1x_serving.tar.gz >/dev/null 2>&1
  tar xf cascade_rcnn_r50_fpx_1x_serving.tar.gz
  sed -i "13s/9292/8879/g" test_client.py
  python3 -m paddle_serving_server_gpu.serve --model serving_server --port 8879 --gpu_id 0 > rcnn_rpc 2>&1 &
  sleep 5
  python3 test_client.py
  check_result $FUNCNAME
  kill_server_process
}

function deeplabv3_rpc() {
  setproxy
  run_gpu_env
  cd ${build_path}/python/examples/deeplabv3
  python3 -m paddle_serving_app.package --get_model deeplabv3 >/dev/null 2>&1
  tar -xzvf deeplabv3.tar.gz >/dev/null 2>&1
  sed -i "22s/9494/8880/g" deeplabv3_client.py
  python3 -m paddle_serving_server_gpu.serve --model deeplabv3_server --gpu_ids 0 --port 8880 > deeplab_rpc 2>&1 &
  sleep 5
  python3 deeplabv3_client.py
  kill_server_process
}

function mobilenet_rpc() {
  setproxy
  run_gpu_env
  cd ${build_path}/python/examples/mobilenet
  python3 -m paddle_serving_app.package --get_model mobilenet_v2_imagenet >/dev/null 2>&1
  tar -xzvf mobilenet_v2_imagenet.tar.gz >/dev/null 2>&1
  sed -i "22s/9393/8881/g" mobilenet_tutorial.py
  python3 -m paddle_serving_server_gpu.serve --model mobilenet_v2_imagenet_model --gpu_ids 0 --port 8881 > mobilenet_rpc 2>&1 &
  sleep 5
  python3 mobilenet_tutorial.py
  kill_server_process
}

function unet_rpc() {
 setproxy
 run_gpu_env
 cd ${build_path}/python/examples/unet_for_image_seg
 python3 -m paddle_serving_app.package --get_model unet >/dev/null 2>&1
 tar -xzvf unet.tar.gz >/dev/null 2>&1
 sed -i "22s/9494/8882/g" seg_client.py
 python3 -m paddle_serving_server_gpu.serve --model unet_model --gpu_ids 0 --port 8882 > haha 2>&1 &
 sleep 5
 python3 seg_client.py
 check_result $FUNCNAME
 kill_server_process
}

function resnetv2_rpc() {
  setproxy
  run_gpu_env
  cd ${build_path}/python/examples/resnet_v2_50
  python3 -m paddle_serving_app.package --get_model resnet_v2_50_imagenet >/dev/null 2>&1
  tar -xzvf resnet_v2_50_imagenet.tar.gz >/dev/null 2>&1
  sed -i 's/9393/8883/g' resnet50_v2_tutorial.py
  python3 -m paddle_serving_server_gpu.serve --model resnet_v2_50_imagenet_model --gpu_ids 0 --port 8883 > v2_log 2>&1 &
  sleep 10
  python3 resnet50_v2_tutorial.py
  kill_server_process
}

function ocr_rpc() {
  setproxy
  run_cpu_env
  cd ${build_path}/python/examples/ocr
  python3 -m paddle_serving_app.package --get_model ocr_rec >/dev/null 2>&1
  tar -xzvf ocr_rec.tar.gz >/dev/null 2>&1
  sed -i 's/9292/8884/g' test_ocr_rec_client.py
  python3 -m paddle_serving_server.serve --model ocr_rec_model --port 8884 > ocr_rpc 2>&1 &
  sleep 5
  python3 test_ocr_rec_client.py
  kill_server_process
}

function fit_a_line_http() {
  unsetproxy
  run_cpu_env
  cd ${build_path}/python/examples/fit_a_line
  python3 -m paddle_serving_server.serve --model uci_housing_model --thread 10 --port 8871 --name uci > http_log 2>&1 &
  sleep 10
  curl -H "Content-Type:application/json" -X POST -d '{"feed":[{"x": [0.0137, -0.1136, 0.2553, -0.0692, 0.0582, -0.0727, -0.1583, -0.0584, 0.6283, 0.4919, 0.1856, 0.0795, -0.0332]}], "fetch":["price"]}' http://${host}:8871/uci/prediction
  check_result $FUNCNAME
  kill_server_process
}

function lac_http() {
  unsetproxy
  run_cpu_env
  cd ${build_path}/python/examples/lac
  python3 lac_web_service.py lac_model/ lac_workdir 8872 > http_lac_log 2>&1 &
  sleep 10
  curl -H "Content-Type:application/json" -X POST -d '{"feed":[{"words": "我爱北京天安门"}], "fetch":["word_seg"]}' http://${host}:8872/lac/prediction
  check_result $FUNCNAME
  kill_server_process
}

function cnn_http() {
  unsetproxy
  run_cpu_env
  cd ${build_path}/python/examples/imdb
  python3 text_classify_service.py imdb_cnn_model/ workdir/ 8873 imdb.vocab > cnn_http 2>&1 &
  sleep 10
  curl -H "Content-Type:application/json" -X POST -d '{"feed":[{"words": "i am very sad | 0"}], "fetch":["prediction"]}' http://${host}:8873/imdb/prediction
  check_result $FUNCNAME
  kill_server_process
}

function bow_http() {
  unsetproxy
  run_cpu_env
  cd ${build_path}/python/examples/imdb
  python3 text_classify_service.py imdb_bow_model/ workdir/ 8874 imdb.vocab > bow_http 2>&1 &
  sleep 10
  curl -H "Content-Type:application/json" -X POST -d '{"feed":[{"words": "i am very sad | 0"}], "fetch":["prediction"]}' http://${host}:8874/imdb/prediction
  check_result $FUNCNAME
  kill_server_process
}

function lstm_http() {
  unsetproxy
  run_cpu_env
  cd ${build_path}/python/examples/imdb
  python3 text_classify_service.py imdb_bow_model/ workdir/ 8875 imdb.vocab > bow_http 2>&1 &
  sleep 10
  curl -H "Content-Type:application/json" -X POST -d '{"feed":[{"words": "i am very sad | 0"}], "fetch":["prediction"]}' http://${host}:8875/imdb/prediction
  check_result $FUNCNAME
  kill_server_process
}

function ResNet50_http() {
  unsetproxy
  run_gpu_env
  cd ${build_path}/python/examples/imagenet
  python3 resnet50_web_service.py ResNet50_vd_model gpu 8876 > resnet50_http 2>&1 &
  sleep 10
  curl -H "Content-Type:application/json" -X POST -d '{"feed":[{"image": "https://paddle-serving.bj.bcebos.com/imagenet-example/daisy.jpg"}], "fetch": ["score"]}' http://${host}:8876/image/prediction
  check_result $FUNCNAME
  kill_server_process
}

function bert_http(){
  unsetproxy
  run_gpu_env
  cd ${build_path}/python/examples/bert
  cp data-c.txt.1 data-c.txt
  cp vocab.txt.1 vocab.txt
  export CUDA_VISIBLE_DEVICES=0
  python3 bert_web_service.py bert_seq128_model/ 8878 > bert_http 2>&1 &
  sleep 5
  curl -H "Content-Type:application/json" -X POST -d '{"feed":[{"words": "hello"}], "fetch":["pooled_output"]}' http://127.0.0.1:8878/bert/prediction
  check_result $FUNCNAME
  kill_server_process
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

function run_http_models(){
  for model in ${http_model_list[@]}
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
  run_http_models
  end_hook
}


main$@
