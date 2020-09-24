#!/usr/bin/env bash
fleetx_path=/workspace/FleetX
version=`date -d @$(git log -1 --pretty=format:%ct) "+%Y%m%d"`
fleet_cpu_model_list=(ctr_app w2v)
fleet_gpu_model_list=(resnet_app vgg_app bert_app transformer_app)
fleet_test_models=(ctr_app w2v)


function setproxy(){
  export http_proxy=${proxy}
  export https_proxy=${proxy}
}


function unsetproxy(){
  unset http_proxy
  unset https_proxy
}


function kill_fleetx_process(){
  kill `ps -ef|grep python|awk '{print $2}'`
}


function check_result() {
    if [ $? -ne 0 ];then
      echo -e "\033[4;31;42m$1 model runs failed, please check your pull request or modify test case! \033[0m"
      exit 1
    else
      echo -e "\033[4;37;42m$1 model runs successfully, congratulations! \033[0m"
    fi
}


function before_hook() {
    wget --no-check-certificate https://fleet.bj.bcebos.com/test/loader/fleet_x-0.0.4-py2-none-any.whl
    pip install fleet_x-0.0.4-py2-none-any.whl
    echo "fleetx installed succ"

    wget http://yq01-gpu-255-125-21-00.epc.baidu.com:8011/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
    pip install paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
    echo "paddlepaddle installed succ"
}


function resnet_single() {
    cd ${fleetx_path}/examples
    sed -i "s/(2)/(1)/g" resnet_single.py
    fleetrun --gpus 0 resnet_single.py
    check_result $FUNCNAME
    kill_fleetx_process
}


function resnet_app() {
    cd ${fleetx_path}/examples
    fleetrun --gpus 0,1 resnet_app.py
    check_result $FUNCNAME
    kill_fleetx_process
}


function vgg_app() {
    cd ${fleetx_path}/examples
    fleetrun --gpus 0,1 vgg_app.py
    check_result $FUNCNAME
    kill_fleetx_process
}



function bert_app() {
    cd ${fleetx_path}/examples
    fleetrun --gpus 0,1 bert_app.py
    check_result $FUNCNAME
    kill_fleetx_process
}


function transformer_app() {
    cd ${fleetx_path}/examples
    fleetrun --gpus 0,1 transformer_app.py
    check_result $FUNCNAME
    kill_fleetx_process
}


function ctr_app() {
    cd ${fleetx_path}/examples
    sed -i "s/epoch=10/epoch=1/g" ctr_app.py
    sed -i "s/ctr_data/train_data/raw_data/g" ctr_app.py
    ln -s /root/.cache/dist_data/serving/criteo_ctr_with_cube/raw_data ./
    fleetrun --gpus 0,1 ctr_app.py
    check_result $FUNCNAME
    kill_fleetx_process
}


function w2v() {
    cd ${fleetx_path}/examples
    fleetrun --gpus 0,1 word2vec_app.py
    check_result $FUNCNAME
    kill_fleetx_process
}


function run_cpu_models(){
      for model in ${fleet_cpu_model_list[@]}
      do
        echo "===========${model} run begin==========="
        $model
        sleep 3
        echo "===========${model} run  end ==========="
      done
}

function run_gpu_models(){
      for model in ${fleet_gpu_model_list[@]}
      do
        echo "===========${model} run begin==========="
        $model
        sleep 3
        echo "===========${model} run  end ==========="
      done
}

function run_test_models(){
      for model in ${fleet_test_models[@]}
      do
        echo "===========${model} run begin==========="
        $model
        sleep 3
        echo "===========${model} run  end ==========="
      done
}

function end_hook(){
  cd ${fleetx_path}/examples
  rm -rf *.data
  rm -rf *.tar.gz
  echo "===========files==========="
  ls -hlst
  echo "=========== end ==========="
}

main() {
    before_hook
    run_test_models
#    run_cpu_models
#    run_gpu_models
#    end_hook
}


main$@
