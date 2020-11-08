function run_bert_large() {
    echo "bert large begin run"
    bert_large=${benchmark_path}/FleetX/2.0benchmark/collective/bert_large
    echo 'run bert_large_fp16_n1c1'
    cd ${bert_large}/n1c1_fp16
    rm -rf bert_benchmark.py
    cp ../bert_benchmark_fp16.py bert_benchmark.py
    fleetsub -f bert_large.yaml

    echo 'run bert_large_fp32_n1c1'
    cd ${bert_large}/n1c1_fp32
    rm -rf bert_benchmark.py
    cp ../bert_benchmark_fp32.py bert_benchmark.py
    fleetsub -f bert_large.yaml

    echo 'run bert_large_fp16_n1c8'
    cd ${bert_large}/n1c8_fp16
    rm -rf bert_benchmark.py
    cp ../bert_benchmark_fp16.py bert_benchmark.py
    fleetsub -f bert_large.yaml

    echo 'run bert_large_fp32_n1c8'
    cd ${bert_large}/n1c8_fp32
    rm -rf bert_benchmark.py
    cp ../bert_benchmark_fp32.py bert_benchmark.py
    fleetsub -f bert_large.yaml

    echo 'run bert_large_fp16_n2c16'
    cd ${bert_large}/n2c16_fp16
    rm -rf bert_benchmark.py
    cp ../bert_benchmark_fp16.py bert_benchmark.py
    fleetsub -f bert_large.yaml

    echo 'run bert_large_fp32_n2c16'
    cd ${bert_large}/n2c16_fp32
    rm -rf bert_benchmark.py
    cp ../bert_benchmark_fp32.py bert_benchmark.py
    fleetsub -f bert_large.yaml

    #30 min after
    sleep 1200

    echo 'run bert_large_fp16_n4c32'
    cd ${bert_large}/n4c32_fp16
    rm -rf bert_benchmark.py
    cp ../bert_benchmark_fp16.py bert_benchmark.py
    fleetsub -f bert_large.yaml

    #20 min after
    sleep 1200

    echo 'run bert_large_fp32_n4c32'
    cd ${bert_large}/n4c32_fp32
    rm -rf bert_benchmark.py
    cp ../bert_benchmark_fp32.py bert_benchmark.py
    fleetsub -f bert_large.yaml

    echo "bert large jobs submit over"
}

run_bert_large
