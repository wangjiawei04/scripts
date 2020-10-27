function run() {
    echo "vgg16 begin run"
    vgg_path=${benchmark_path}/FleetX/2.0benchmark/collective/vgg
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
    sleep 1200

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

run