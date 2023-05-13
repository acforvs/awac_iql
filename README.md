## Offline RL: AWAC & IQL

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Black](https://img.shields.io/badge/code%20style-black-000000.svg)

This repository contains a PyTorch implementation of two offline reinforcement learning algorithms:
1. AWAC: Accelerating Online Reinforcement Learning with Offline Datasets
2. Offline Reinforcement Learning with Implicit Q-Learning


### Installation

Each algorithm is provided in two versions: a `.py` file and an `.ipynb` notebook. It is recommended to run the IPython versions directly in Google Colab. For the `.py` file to work, you need to install and set up `d4rl` and `mujoco` on your local machine. You can use the following command for installation:
```bash
 apt-get -qq update
 apt-get -qq install -y libosmesa6-dev libgl1-mesa-glx libglfw3 libgl1-mesa-dev libglew-dev patchelf
 mkdir ~/.mujoco
 wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz
 tar -zxf mujoco.tar.gz -C "$HOME/.mujoco"
 rm mujoco.tar.gz
 echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc
 echo 'export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so' >> ~/.bashrc
 echo "/root/.mujoco/mujoco210/bin" > /etc/ld.so.conf.d/mujoco_ld_lib_path.conf
 ldconfig
 pip3 install -U 'mujoco-py<2.2,>=2.1'
 touch .mujoco_setup_complete
```
Please note that we cannot guarantee that this script will successfully install all the required libraries on your local machine. If any values are missing in your `~/.bashrc` file, an error will be reported upon launching.

To set up your local machine, you can also consider using the Dockerfile available at: https://github.com/tinkoff-ai/CORL/blob/main/Dockerfile

### Training

To run training locally, execute the following command:
```
python3 iql.py --yaml_path=configs/iql/antmaze_medium_seed_0.yaml
```
Ensure that the `yaml_path` parameter contains the path to the `.yaml` file with the desired configuration.


### Credits

This implementation draws significant inspiration from the following repositories:
* https://github.com/tinkoff-ai/CORL 
* https://github.com/ikostrikov/jaxrl
* https://github.com/ikostrikov/implicit_q_learning
* https://github.com/rail-berkeley/rlkit/tree/master/examples/iql
* https://github.com/Manchery/iql-pytorch/tree/master
