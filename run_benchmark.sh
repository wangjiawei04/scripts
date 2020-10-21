function prepare() {
  cd ${benchmark_path}
  if [[ -d scripts ]]; then
    rm -rf scripts FleetX
  fi
  git clone ssh://g@gitlab.baidu.com:8022/liyang109/FleetX.git
  git clone https://github.com/gentelyang/scripts.git
}

function run_vgg() {
    echo "vgg16 begin run"
    vgg_path=${benchmark_path}/FleetX/collective/vgg
    echo 'run vgg16_fp16_n1c1'
    cd ${vgg_path}/n1c1_fp16
    rm -rf vgg_benchmark.py
    cp ../vgg_benchmark_fp16.py vgg_benchmark.py
    fleetsub -f vgg.yaml

    echo 'run vgg16_fp32_n1c1'
    cd ${vgg_path}/n1c1_fp32
    rm -rf vgg_benchmark.py
    cp ../vgg_benchmark_fp32.py vgg_benchmark.py
    fleetsub -f vgg.yaml

    echo 'run vgg16_fp16_n1c8'
    cd ${vgg_path}/n1c8_fp16
    rm -rf vgg_benchmark.py
    cp ../vgg_benchmark_fp16.py vgg_benchmark.py
    fleetsub -f vgg.yaml

    echo 'run vgg16_fp32_n1c8'
    cd ${vgg_path}/n1c8_fp32
    rm -rf vgg_benchmark.py
    cp ../vgg_benchmark_fp32.py vgg_benchmark.py
    fleetsub -f vgg.yaml

    echo 'run vgg16_fp16_n2c16'
    cd ${vgg_path}/n2c16_fp16
    rm -rf vgg_benchmark.py
    cp ../vgg_benchmark_fp16.py vgg_benchmark.py
    fleetsub -f vgg.yaml

    echo 'run vgg16_fp32_n2c16'
    cd ${vgg_path}/n2c16_fp32
    rm -rf vgg_benchmark.py
    cp ../vgg_benchmark_fp32.py vgg_benchmark.py
    fleetsub -f vgg.yaml

    #30 min after
    sleep 1800

    echo 'run vgg16_fp16_n4c32'
    cd ${vgg_path}/n4c32_fp16
    rm -rf vgg_benchmark.py
    cp ../vgg_benchmark_fp16.py vgg_benchmark.py
    fleetsub -f vgg.yaml

    #20 min after
    sleep 1200

    echo 'run vgg16_fp32_n4c32'
    cd ${vgg_path}/n4c32_fp32
    rm -rf vgg_benchmark.py
    cp ../vgg_benchmark_fp32.py vgg_benchmark.py
    fleetsub -f vgg.yaml

    echo "vgg16 jobs submit over"
}


function run_resnet() {
    echo "resnet begin run"
    resnet_path=${benchmark_path}/FleetX/collective/resnet
    echo 'run resnet_fp16_n1c1'
    cd ${resnet_path}/n1c1_fp16
    rm -rf resnet_benchmark.py
    cp ../resnet_benchmark_fp16.py resnet_benchmark.py
    fleetsub -f resnet.yaml

    echo 'run resnet_fp32_n1c1'
    cd ${resnet_path}/n1c1_fp32
    rm -rf resnet_benchmark.py
    cp ../resnet_benchmark_fp32.py resnet_benchmark.py
    fleetsub -f resnet.yaml

    echo 'run resnet_fp16_n1c8'
    cd ${resnet_path}/n1c8_fp16
    rm -rf resnet_benchmark.py
    cp ../resnet_benchmark_fp16.py resnet_benchmark.py
    fleetsub -f resnet.yaml

    echo 'run resnet_fp32_n1c8'
    cd ${resnet_path}/n1c8_fp32
    rm -rf resnet_benchmark.py
    cp ../resnet_benchmark_fp32.py resnet_benchmark.py
    fleetsub -f resnet.yaml

    echo 'run resnet_fp16_n2c16'
    cd ${resnet_path}/n2c16_fp16
    rm -rf resnet_benchmark.py
    cp ../resnet_benchmark_fp16.py resnet_benchmark.py
    fleetsub -f resnet.yaml

    echo 'run resnet_fp32_n2c16'
    cd ${resnet_path}/n2c16_fp32
    rm -rf resnet_benchmark.py
    cp ../resnet_benchmark_fp32.py resnet_benchmark.py
    fleetsub -f resnet.yaml

    #30 min after
    sleep 1800

    echo 'run resnet_fp16_n4c32'
    cd ${resnet_path}/n4c32_fp16
    rm -rf resnet_benchmark.py
    cp ../resnet_benchmark_fp16.py resnet_benchmark.py
    fleetsub -f resnet.yaml

    #20 min after
    sleep 1200

    echo 'run resnet_fp32_n4c32'
    cd ${resnet_path}/n4c32_fp32
    rm -rf resnet_benchmark.py
    cp ../resnet_benchmark_fp32.py resnet_benchmark.py
    fleetsub -f resnet.yaml

    echo "resnet50 jobs submit over"
}


function run_transformer() {
    echo "transformer begin run"
    transformer_app=${benchmark_path}/FleetX/collective/trans
    echo 'run transformer_fp16_n1c1'
    cd ${transformer_app}/n1c1_fp16
    rm -rf transformer_app.py
    cp ../transformer_app_fp16.py transformer_app.py
    fleetsub -f transformer_app.yaml

    echo 'run transformer_fp32_n1c1'
    cd ${transformer_app}/n1c1_fp32
    rm -rf transformer_app.py
    cp ../transformer_app_fp32.py transformer_app.py
    fleetsub -f transformer_app.yaml

    echo 'run transformer_fp16_n1c8'
    cd ${transformer_app}/n1c8_fp16
    rm -rf transformer_app.py
    cp ../transformer_app_fp16.py transformer_app.py
    fleetsub -f transformer_app.yaml

    echo 'run transformer_fp32_n1c8'
    cd ${transformer_app}/n1c8_fp32
    rm -rf transformer_app.py
    cp ../transformer_app_fp32.py transformer_app.py
    fleetsub -f transformer_app.yaml

    echo 'run transformer_fp16_n2c16'
    cd ${transformer_app}/n2c16_fp16
    rm -rf transformer_app.py
    cp ../transformer_app_fp16.py transformer_app.py
    fleetsub -f transformer_app.yaml

    echo 'run transformer_fp32_n2c16'
    cd ${transformer_app}/n2c16_fp32
    rm -rf transformer_app.py
    cp ../transformer_app_fp32.py transformer_app.py
    fleetsub -f transformer_app.yaml

    #30 min after
    sleep 1800

    echo 'run transformer_fp16_n4c32'
    cd ${transformer_app}/n4c32_fp16
    rm -rf transformer_app.py
    cp ../transformer_app_fp16.py transformer_app.py
    fleetsub -f transformer_app.yaml

    #20 min after
    sleep 1200

    echo 'run transformer_fp32_n4c32'
    cd ${transformer_app}/n4c32_fp32
    rm -rf transformer_app.py
    cp ../transformer_app_fp32.py transformer_app.py
    fleetsub -f transformer_app.yaml

    echo "transformer jobs submit over"
}

prepare
run_vgg
