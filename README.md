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

## General interface
- The python interface for calling the FPGA's denoising service on the server.

## Deployment
- Deploy the serve app and FPGA docker enviroment following "general interface" part.
- Here we take centos + Alveo U50 as a typical example.
- Input: xmodel, dataset.
- Ouput: benchmarking result.

## Reference and resources
- Float model dev version: https://github.com/MrBled/HardwareDenoiser
- PTQ+QAT flow dev version: https://github.com/CNStanLee/Emerald_Video_Denoise_Acc
- pix2pix: https://github.com/phillipi/pix2pix
- BSD68: https://www.kaggle.com/code/mpwolke/berkeley-segmentation-dataset-68
- BSD68C: https://github.com/clausmichele/CBSD68-dataset?tab=readme-ov-file
- BSD100: https://www.kaggle.com/datasets/asilva1691/bsd100
- SET12: https://www.kaggle.com/datasets/leweihua/set12-231008
- Uploaded pre-processed dataset for your benchmarking convenience: https://drive.google.com/drive/folders/16fbUAJ0pD-zlKcj4Qxqu7_LQmG7Rp2oo?usp=sharing


## Citation
- If you found this work is useful for you, please cite our work below.
``` bib
@article{li2025retide,
  title={ReTiDe: Real-Time Denoising for Energy-Efficient Motion Picture Processing with FPGAs},
  author={Li, Changhong and Bled, Cl{\'e}ment and Fernandez, Rosa and Shanker, Shreejith},
  journal={arXiv preprint arXiv:2510.03812},
  year={2025}
}
```

## Demo
- A demo video could be found at: https://www.youtube.com/watch?v=0epcrRA_f2w

## Email
- For any technical discussion or deployment issues, don't be hesitate to email us :) lic9@tcd.ie.

## Acknowledgement
This work was funded by the Horizon CL4 2022 - EU Project Emerald – 101119800.