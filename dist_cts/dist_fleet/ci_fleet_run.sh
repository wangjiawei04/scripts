#!/bin/bash

function prepare(){
    env
    ln -s /ssd3/ly/cts_ce/dataset/data/* ./data/
    export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.15-ucs4/lib/:/ssd2/work/418.39/lib64/:${LD_LIBRARY_PATH}
    export PATH=/opt/_internal/cpython-2.7.15-ucs4/bin/:${PATH}
    pip uninstall paddlepaddle-gpu -y
    pip uninstall paddlepaddle -y
    pip install ${IMAGE_NAME}
    pip install nose
    pip install nose-html-reporting
    unset http_proxy
    unset https_proxy
    echo "IMAGE_NAME is: ${IMAGE_NAME}, RUN_FREQUENCY is: ${RUN_FREQUENCY}"
}

function run(){
    if [[ ${DEVICE_TYPE} == 'GPU' ]]; then
        cases="test_dist_fleet_resnet.py \
               test_dist_fleet_vgg.py"
#               test_dist_fleet_bert.py \
#               test_dist_fleet_transformer.py \
#               test_dist_fleet_launch.py"
    elif [[ ${DEVICE_TYPE} == 'CPU' && ${PSLIB} == 'OFF' ]]; then
        cases="test_dist_fleet_ctr.py \
               test_dist_fleet_deepFM.py \
               test_dist_fleet_save_persistables.py \
               test_dist_fleet_infer.py"
    elif [[ ${DEVICE_TYPE} == 'CPU' && ${PSLIB} == 'ON' ]]; then
        cases="test_dist_fleet_pslib.py"

    else
        echo "NOT SUPPORT"
    fi

    for file in ${cases}
    do
        echo ${file}
        nosetests -s -v --with-html --html-report=${LOG_PATH}/${file}.html ${file}
    done
}

prepare
run
