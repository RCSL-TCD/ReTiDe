# General interface
## Req
- Vitis-AI 1.4.1: could git from official github, the original docker image is expired, we mannully target to: xilinx/viti-ai-cpu:1.4.1.978
- The model used in this part please refer to /released_model.
- The data used in this part please refer to main readme: uploaded pre-processed data.
## How to run this
- (bash): basic enviroment on the server.
- (vai): vitis-ai docker on the server.
- (client): bash on the client side.
## Server
- Ensure your firewall setup, for our instance, we use port 5000.
```shell
(bash) sudo systemctl stop firewalld
(bash) sudo systemctl start firewalld
(bash) sudo firewall-cmd --permanent --add-port=5000/tcp
(bash) sudo firewall-cmd --reload
```
- Activate the docker image on the server.
```shell
(bash) cd /home/emurphy/changhong_workspace/Vitis-AI
(bash) source start_vai_141.sh
```
- In the vitis-ai docker, activate the DPU, you must activate it everytime you start the docker 
```shell
(vai) conda activate vitis-ai-pytorch
(vai) source /workspace/setup/alveo/setup.sh DPUCAHX8H
(vai) pip install flask
```
- Run the server
```shell
(vai) cd /mnt/data6/prj/ReTiDe/general_interface/server
(vai) python app.py
```
## Client
- Firstly check configs/server_config.py, change the server ip to your server and corresponding port, for this demo instance, it is:
```shell
server_ip = "134.226.86.156"
port = 5000
```
- Run client.py, this will demo gpu and fpga's single and multiple inference on the server.
```shell
(client) python client.py
```
- Observe the log both on the client and server.
```shell
(client) ✓ upload ok
(client) save on server: noisy_0_20250929_054545_68b66ca9.png
(client)   prediction result: processed_image
(client) ✓ processed image saved: results/images/processed_noisy_0_20250929_124838.png
(client) ✓ original copy saved: results/images/original_noisy_0_20250929_124838.png
(client) ✓ complete result saved: results/result_noisy_0_20250929_124838.json
```