conda activate vitis-ai-pytorch

# folders
export BUILD=./build
export LOG=${BUILD}/logs
mkdir -p ${LOG}


source compile_ptq_old.sh u50 ${BUILD} ${LOG} --debug
source compile_qat_old.sh u50 ${BUILD} ${LOG} --debug


# python -u target.py --target u50 --num_images 10    -d ${BUILD} 2>&1 | tee ${LOG}/target_u50.log

