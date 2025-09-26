# Vitis-AI UNet

## Purpose
This project is to rebuild Emmet Murphy's workflow aka transfering a UNet to Alveo U50 for denosing

## Structure
- model_for_training: float models for training
- release_deployable: released xmodel and drivers verified with Alveo U50 on the server
- vai_unet: workspace and source code to build the protypes and xmodels, main work path is 5_vai_container.

## Preparation
Emmet's project was built up using Vitis-AI 1.4.1,
we dont know if things build with this version will happy with the things built with VAI 3.0/3.5, thus we use VAI 1.4.1 to rebuild the flow.
so we need to config things below:
### Vitis 2021.1
VAI 1.4.1 only work with Vitis 2021.1
### VAI 1.4.1
clone from VAI github and switch to 1.4.1
### Docker Image
cd /home/changhong/prj/vai_unet/0_follow_tutorial/09-mnist_pyt/files
./docker_run.sh xilinx/vitis-ai-cpu:1.4.1.978
We'd better to mannuly choose vai version to avoid any version problems

## How to run

```shell
Vitis-AI /workspace > conda activate vitis-ai-pytorch
(vitis-ai-pytorch) Vitis-AI /workspace >
```

```shell
(vitis-ai-pytorch) Vitis-AI /workspace > source run_all.sh
```

