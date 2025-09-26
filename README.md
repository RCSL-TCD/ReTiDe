# ReTiDe-Real-Time-Denoising-for-Energy-Efficient-Motion-Picture-Processing-with-FPGAs
- Code of paper: ReTiDe: Real-Time Denoising for Energy-Efficient Motion Picture Processing with FPGAs

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
- Input: xmodel, dataset.
- Ouput: benchmarking result.
## General interface
- The python interface for calling the FPGA's denoising service on the server.

## Reference
- Float model dev version: https://github.com/MrBled/HardwareDenoiser
- PTQ+QAT flow dev version: https://github.com/CNStanLee/Emerald_Video_Denoise_Acc
- pix2pix: https://github.com/phillipi/pix2pix

## Citation
- update our arxiv link here.

## Demo
- A demo video could be found at: https://www.youtube.com/watch?v=0epcrRA_f2w

## Acknowledgement
This work was funded by the Horizon CL4 2022 - EU Project Emerald – 101119800.