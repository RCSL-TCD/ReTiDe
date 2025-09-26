conda activate vitis-ai-pytorch

export BUILD=./build
export LOG=${BUILD}/logs
mkdir -p ${LOG}

source compile_ptq.sh u50 ${BUILD} ${LOG} --debug
source compile_qat.sh u50 ${BUILD} ${LOG} --debug



