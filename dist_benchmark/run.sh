function prepare() {
  cd ${benchmark_path}
  if [[ -d scripts ]]; then
    rm -rf scripts FleetX
  fi
  git clone ssh://g@gitlab.baidu.com:8022/liyang109/FleetX.git
  git clone https://github.com/gentelyang/scripts.git

}
function run_benchmark() {
  cd ${benchmark_path}/scripts/dist_benchmark
  sh run_vgg.sh > job_vgg_log
  sleep 1200
  sh run_r50.sh > job_r50_log
  sleep 1200
  sh run_trans.sh > job_trans_log
  sleep 1200
  sh run_bert_base.sh > job_bert_base_log
  sleep 1200
  sh run_bert_base128.sh > job_bert_base128_log
  sleep 1200
  sh run_bert_large.sh > job_bert_large_log
  sleep 1200
  sh run_bert_large128.sh > job_bert_large128_log
  echo "all job submit succ"
}


prepare
run_benchmark
