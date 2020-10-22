function run_transformer() {
    echo "transformer begin run"
    transformer_app=${benchmark_path}/FleetX/2.0benchmark/collective/trans
    echo 'run transformer_fp16_n1c1'
    cd ${transformer_app}/n1c1_fp16
    rm -rf transformer_app.py
    cp ../transformer_app_fp16.py transformer_app.py
    fleetsub -f trans.yaml

    echo 'run transformer_fp32_n1c1'
    cd ${transformer_app}/n1c1_fp32
    rm -rf transformer_app.py
    cp ../transformer_app_fp32.py transformer_app.py
    fleetsub -f trans.yaml

    echo 'run transformer_fp16_n1c8'
    cd ${transformer_app}/n1c8_fp16
    rm -rf transformer_app.py
    cp ../transformer_app_fp16.py transformer_app.py
    fleetsub -f trans.yaml

    echo 'run transformer_fp32_n1c8'
    cd ${transformer_app}/n1c8_fp32
    rm -rf transformer_app.py
    cp ../transformer_app_fp32.py transformer_app.py
    fleetsub -f trans.yaml

    echo 'run transformer_fp16_n2c16'
    cd ${transformer_app}/n2c16_fp16
    rm -rf transformer_app.py
    cp ../transformer_app_fp16.py transformer_app.py
    fleetsub -f trans.yaml

    echo 'run transformer_fp32_n2c16'
    cd ${transformer_app}/n2c16_fp32
    rm -rf transformer_app.py
    cp ../transformer_app_fp32.py transformer_app.py
    fleetsub -f trans.yaml

    #30 min after
    sleep 1200

    echo 'run transformer_fp16_n4c32'
    cd ${transformer_app}/n4c32_fp16
    rm -rf transformer_app.py
    cp ../transformer_app_fp16.py transformer_app.py
    fleetsub -f trans.yaml

    #20 min after
    sleep 1200

    echo 'run transformer_fp32_n4c32'
    cd ${transformer_app}/n4c32_fp32
    rm -rf transformer_app.py
    cp ../transformer_app_fp32.py transformer_app.py
    fleetsub -f trans.yaml

    echo "transformer jobs submit over"
}

run_transformer