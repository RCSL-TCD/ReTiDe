
## How to deploy on Alveo server

## Alveo U50

**Note:** The U50 should be flashed with the correct deployment shell, and this should have been done in the 'Preparing the host machine and target boards' section above.

The following steps should be run from inside the Vitis-AI Docker container:

  + Ensure that Vitis-AI's PyTorch conda environment is enabled (if not, the run `conda activate vitis-ai-pytorch`).

  + Run `source setup.sh DPUCAHX8H` which sets environment variables to point to the correct overlay for the U50. The complete steps to run are as follows:


```shell
conda activate vitis-ai-pytorch
source setup.sh DPUCAHX8H
cd build/target_u50
/usr/bin/python3 app_mt.py -m CNN_u50.xmodel
```

The console output will be like this:

```shell
(vitis-ai-pytorch) Vitis-AI /workspace/build/target_u50 > /usr/bin/python3 app_mt.py -m CNN_u50.xmodel
Command line options:
 --image_dir :  images
 --threads   :  1
 --model     :  CNN_u50.xmodel
-------------------------------
Pre-processing 10000 images...
-------------------------------
Starting 1 threads...
-------------------------------
Throughput=14362.58 fps, total frames = 10000, time=0.6963 seconds
Correct:9877, Wrong:123, Accuracy:0.9877
-------------------------------
```

Perfromance can be slightly increased by increasing the number fo threads:

```shell
(vitis-ai-pytorch) Vitis-AI /workspace/build/target_u50 > /usr/bin/python3 app_mt.py -m CNN_u50.xmodel -t 6
Command line options:
 --image_dir :  images
 --threads   :  6
 --model     :  CNN_u50.xmodel
-------------------------------
Pre-processing 10000 images...
-------------------------------
Starting 6 threads...
-------------------------------
Throughput=16602.34 fps, total frames = 10000, time=0.6023 seconds
Correct:9877, Wrong:123, Accuracy:0.9877
-------------------------------
```

## Deployable

Please find the deployable model in /release_deployable