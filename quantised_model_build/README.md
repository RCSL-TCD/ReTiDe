# Vitis-AI UNet

## Purpose
This project is to build a vitis-ai UNet to Alveo U50 for denoising

## Preparation
Emmet's project was built up using Vitis-AI 1.4.1,
we dont know if things build with this version will happy with the things built with VAI 3.0/3.5, thus we use VAI 1.4.1 to rebuild the flow.
so we need to config things below:
### Vitis 2021.1
VAI 1.4.1 only work with Vitis 2021.1
### VAI 1.4.1
clone from VAI github and switch to 1.4.1
### Docker Image
We have 3 docker images which could be downloaded and run with three scripts.
- start_vai.sh: vitis-ai-cpu:1.4.1.978, for xmodel build flow as it is the last version support Alveo U50 DPU.
- start_vai_25gpu.sh: vitis-ai-gpu:latest, for gpu QAT.
- start_vai_3.sh :vitis-ai:latest, if you wanna try more advanced DPUs.

## How to run

```shell
(bash) source start_vai.sh
```shell

```shell
Vitis-AI /workspace > conda activate vitis-ai-pytorch
(vitis-ai-pytorch) Vitis-AI /workspace >
```

```shell
(vitis-ai-pytorch) Vitis-AI /workspace > source run_all.sh
```

