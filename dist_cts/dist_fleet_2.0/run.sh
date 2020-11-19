#!/bin/bash

function prepare(){
    env
    ln -s ../dist_fleet/thirdparty ./
    ln -s /ssd3/ly/cts_ce/dataset/data/ ./
    export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.15-ucs4/lib/:/ssd2/work/418.39/lib64/:${LD_LIBRARY_PATH}
    export PATH=/opt/_internal/cpython-2.7.15-ucs4/bin/:${PATH}
    pip uninstall paddlepaddle-gpu -y
    pip uninstall paddlepaddle -y
    pip install ${all_path}/images/${IMAGE_NAME}
    echo "paddlepaddle install succ"
    pip install nose
    pip install nose-html-reporting
    unset http_proxy
    unset https_proxy
    echo "IMAGE_NAME is: ${IMAGE_NAME}, RUN_FREQUENCY is: ${RUN_FREQUENCY}"
}

function run(){
    cases="test_dist_fleet_static_vgg.py \
           test_dist_fleet_dygraph_api.py \
           test_dist_fleet_static_strategy.py \
           test_dist_fleet_static_launch.py \
           test_dist_fleet_static_fleetrun.py \
           test_dist_fleet_dygraph_gloo_init.py \
           test_dist_fleet_static_ctr.py "
    for file in ${cases}
    do
        echo ${file}
        nosetests -s -v  ${file}
    done
}

prepare
run
