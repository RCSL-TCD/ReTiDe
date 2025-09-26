# ReTiDe-Real-Time-Denoising-for-Energy-Efficient-Motion-Picture-Processing-with-FPGAs
- Code of papser: ReTiDe: Real-Time Denoising for Energy-Efficient Motion Picture Processing with FPGAs

## Float model build
- The training flow of the float model.
- Input: model_def, dataset.
- Output: model_pt_file
## Quantised model build
- Model's quantisation including Post-training Quantisation(PTQ) and Quantisation-aware Training(QAT).
- Input: model_pt_file
- Output: xmodel
## Released model
- This contains the model evaluated in the paper, including float32, PTQ, and QAT version.
## Deployment
- Follow this flow to build the server enviroment.
- Here we take centos + Alveo U50 as a typical example.
## General interface
- The python interface for calling the FPGA's denoising service on the server.