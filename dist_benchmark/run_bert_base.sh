function run_bert_base() {
    echo "bert base begin run"
    bert_base=${benchmark_path}/FleetX/2.0benchmark/collective/bert_base
    echo 'run bert_base_fp16_n1c1'
    cd ${bert_base}/n1c1_fp16
    rm -rf bert_benchmark.py
    cp ../bert_benchmark_fp16.py bert_benchmark.py
    fleetsub -f bert_base.yaml

    echo 'run bert_base_fp32_n1c1'
    cd ${bert_base}/n1c1_fp32
    rm -rf bert_benchmark.py
    cp ../bert_benchmark_fp32.py bert_benchmark.py
    fleetsub -f bert_base.yaml

    echo 'run bert_base_fp16_n1c8'
    cd ${bert_base}/n1c8_fp16
    rm -rf bert_benchmark.py
    cp ../bert_benchmark_fp16.py bert_benchmark.py
    fleetsub -f bert_base.yaml

    echo 'run bert_base_fp32_n1c8'
    cd ${bert_base}/n1c8_fp32
    rm -rf bert_benchmark.py
    cp ../bert_benchmark_fp32.py bert_benchmark.py
    fleetsub -f bert_base.yaml

    echo 'run bert_base_fp16_n2c16'
    cd ${bert_base}/n2c16_fp16
    rm -rf bert_benchmark.py
    cp ../bert_benchmark_fp16.py bert_benchmark.py
    fleetsub -f bert_base.yaml

    echo 'run bert_base_fp32_n2c16'
    cd ${bert_base}/n2c16_fp32
    rm -rf bert_benchmark.py
    cp ../bert_benchmark_fp32.py bert_benchmark.py
    fleetsub -f bert_base.yaml

    #30 min after
    sleep 1200

    echo 'run bert_base_fp16_n4c32'
    cd ${bert_base}/n4c32_fp16
    rm -rf bert_benchmark.py
    cp ../bert_benchmark_fp16.py bert_benchmark.py
    fleetsub -f bert_base.yaml

    #20 min after
    sleep 1200

    echo 'run bert_base_fp32_n4c32'
    cd ${bert_base}/n4c32_fp32
    rm -rf bert_benchmark.py
    cp ../bert_benchmark_fp32.py bert_benchmark.py
    fleetsub -f bert_base.yaml

    echo "bert base jobs submit over"
}

run_bert_base
