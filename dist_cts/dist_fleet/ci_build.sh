export FLAGS_fraction_of_gpu_memory_to_use=0.15
export CTEST_OUTPUT_ON_FAILURE=1
export CTEST_PARALLEL_LEVEL=4
export APT_MIRROR=s#http://archive.ubuntu.com/ubuntu#mirror://mirrors.ubuntu.com/mirrors.txt#g
export WITH_GPU=ON
export WITH_CACHE=OFF
export CUDA_ARCH_NAME=Auto
export WITH_AVX=ON
export WITH_MKL=ON
export WITH_TESTING=OFF
export WITH_COVERAGE=OFF
export COVERALLS_UPLOAD=OFF
export GIT_PR_ID=$AGILE_PULL_ID
export JSON_REPO_TOKEN=JSUOs6TF6fD2i30OJ5o2S55V8XWv6euen
export PADDLE_VERSION=0.0.0
export CMAKE_BUILD_TYPE=Release
export PADDLE_FRACTION_GPU_MEMORY_TO_USE=0.15
export CUDA_VISIBLE_DEVICES=0,1
export WITH_DISTRIBUTE=ON
export WITH_ANAKIN=OFF
export RUN_TEST=OFF
export DLE_API_SPEC_URL=https://raw.githubusercontent.com/PaddlePaddle/FluidAPISpec/master/API.spec
export GITHUB_API_TOKEN=5ae37427a95d9e51fb0ea3c71121b75d5c2ebefa
export CHANGED_FILES=paddle/fluid/platform/enforce.h
export BRANCH=develop
export WITH_NGRAPH=ON
export http_proxy=http://172.19.57.45:3128
export https_proxy=http://172.19.57.45:3128
export work_dir="/workspace/Paddle"
export PADDLE_ROOT=$work_dir
export md5_content=$(cat \
            ${work_dir}/cmake/external/*.cmake \
            |md5sum | awk '{print $1}')
xz_dir="/root/third_party/PR-SYS-P0"
xz_file_tar="${xz_dir}/${md5_content}.tar"
xz_file="${xz_dir}/${md5_content}.tar.xz"
bce_file="/home/bce-python-sdk-0.8.33/BosClient.py"


ifconfig
cat /etc/issue


export PYTHON_ABI=cp27-cp27mu
ln -s /usr/lib64/libz.so /usr/local/lib/libz.so
ln -s /usr/local/lib/libnccl.so /usr/local/cuda/lib64/


git config --global user.name "PaddleCI"
git config --global user.email "paddle_ci@example.com"

set +e
#git log  -1 | grep -w "test=document_fix"
#if [ $? == 0 ];then
#  exit 0
#fi


git remote | grep upstream
if [ $? != 0 ]; then git remote add upstream https://github.com/PaddlePaddle/Paddle.git; fi
set -e

git fetch upstream
git checkout -b origin_pr
git checkout -b test_pr upstream/${BRANCH}
git merge --no-edit origin_pr
git log --pretty=oneline -10

rm -rf ${xz_file}

if [ "$WITH_CACHE" == "ON" ];then
    #判断本地有没有cache，没有就去bos拉，cache需要手动更新
    if [ ! -d /root/.cache ];then
        rm -rf /root/cache.tar.gz
        wget -q -P /root --no-proxy --no-check-certificate https://paddle-docker-tar.bj.bcebos.com/home/users/tianshuo/bce-python-sdk-0.8.27/cache.tar.gz
        #tar xf /root/cache.tar.gz
    fi

    #判断本地有没有ccache，没有就去bos拉，cache需要手动更新
    if [ ! -d /root/.ccache ];then
        rm -rf /root/ccache.tar.gz
        wget -q -P /root --no-proxy --no-check-certificate https://paddle-docker-tar.bj.bcebos.com/home/users/tianshuo/bce-python-sdk-0.8.27/ccache.tar.gz
        #tar xf /root/ccache.tar.gz
    fi

    if [ ! -d "$xz_dir" ];then
        mkdir -p ${xz_dir}
    fi

    #判断本地有没有third_party缓存，没有就去bos拉，如果拉下来就使用，没有拉下来就设置update_cached_package=on，执行成功后会判断这个变量为on就会往bos上推送third_party缓存。
    if [ ! -f "${xz_file}" ];then
        set +e
        wget -q --no-proxy --no-check-certificate https://xly-devops.bj.bcebos.com/root/third_party/PR-SYS-P0/${md5_content}.tar.xz
        if [ $? -eq 0 ];then
            set -e
            file_push=ON
            mkdir -p ${work_dir}/build
           # tar xpf ${md5_content}.tar.xz -C ${work_dir}/build
            mv ${md5_content}.tar.xz ${xz_dir}
        else
            set -e
            file_push=ON
            update_cached_package=ON
        fi
    else
        mkdir -p ${work_dir}/build
        # tar xpf ${xz_file} -C ${work_dir}/build
    fi
fi
echo PyGithub >> "${work_dir}/python/requirements.txt"
echo coverage >>"${work_dir}/python/requirements.txt"
echo pycrypto >>"${work_dir}/python/requirements.txt"

wget https://nixos.org/releases/patchelf/patchelf-0.10/patchelf-0.10.tar.gz
tar -zxf patchelf-0.10.tar.gz
cd patchelf-0.10
./configure
make install
cd ..

export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.11-ucs4/lib:${LD_LIBRARY_PATH#/opt/_internal/cpython-2.7.11-ucs2/lib:}
export PATH=/opt/python/cp27-cp27mu/bin/:${PATH}
PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/python/cp27-cp27mu/bin/python -DPYTHON_INCLUDE_DIR:PATH=/opt/python/cp27-cp27mu/include/python2.7 -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-2.7.11-ucs4/lib/libpython2.7.so"
pip install -r ${PADDLE_ROOT}/python/requirements.txt

#./paddle/scripts/paddle_build.sh build


# add by xudong
# 这里不用分成两个任务，当前这种方式不能执行
# 后面统一编译时再考虑即可

./paddle/scripts/paddle_build.sh build_only
pip install /paddle/build/opt/paddle/share/wheels/*.whl
git clone https://github.com/PaddlePaddle/continuous_integration.git
cd continuous_integration/distributed/dist_fleet
pip install nose
pip install nose-html-reporting
unset http_proxy
unset https_proxy
export RUN_FREQUENCY=DAILY

cases="test_dist_fleet_resnet.py \
        test_dist_fleet_save_persistables.py \
        test_dist_fleet_infer.py"

for file in ${cases}
do
    echo ${file}
    nosetests -s -v --with-html --html-report=${LOG_PATH}/${file}.html ${file}
 done