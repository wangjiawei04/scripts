#!/bin/bash
function prepare(){
#    git clone http://github.com//PaddleRec.git
  #  pushd PaddleRec
    cd PaddleRec
    pip3 uninstall paddle-rec -y
    python3 setup.py install
    
    pip3 uninstall paddlepaddle -y
    pip3 install paddlepaddle-gpu==1.7.2.post107
    pip3 install nose
    pip3 install ruamel.yaml
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
