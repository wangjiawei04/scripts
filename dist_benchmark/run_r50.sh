function run_resnet() {
    echo "resnet begin run"
    resnet_path=${benchmark_path}/FleetX/2.0benchmark/collective/resnet
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
    sleep 1200

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


run_resnet