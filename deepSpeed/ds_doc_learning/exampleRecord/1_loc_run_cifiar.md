# deepspeed 单机单卡本地运行 & Docker 运行

本文笔者基于官方示例`DeepSpeedExamples/training/cifar/cifar10_deepspeed.py`进行本地构建和Docker构建运行示例（<font color="darkred">下列代码中均是踩坑后可执行的代码，尤其是Docker部分</font>）, 全部code可以看[笔者的github]()


# 1 环境配置

## 1.1 cuda 相关配置
```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-ubuntu2204-12-3-local_12.3.0-545.23.06-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu2204-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3
```

## 1.2 deepspeed 相关配置

需要注意`pip install mpi4py`可能无法安装，所以用`conda`进行安装
```shell
sudo apt-get update
sudo apt-get install -y openmpi-bin  libopenmpi-dev ninja-build python3-mpi4py numactl
echo "Setting System Param >>>>>"
echo "export PATH=/usr/bin/mpirun:\$PATH" >> ~/.bashrc
echo "export PATH=/usr/bin/mpiexec:\$PATH" >> ~/.bashrc
echo "export PATH=/opt/conda/bin/ninja:\$PATH" >> ~/.bashrc
echo "export PATH=/usr/bin/mpirun:\$PATH" >> ~/.profile
echo "export PATH=/usr/bin/mpiexec:\$PATH" >> ~/.profile
echo "export PATH=/opt/conda/bin/ninja:\$PATH" >> ~/.profile

pip3 install deepspeed==0.12.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install -c conda-forge mpi4py
pip3 install tqdm   -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install triton  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 1.3 Docker相关配置

如果不进行docker运行，这部分可以直接跳过

1. 安装`nvidia-container-toolkit`
```shell
# clear cache
sudo docker builder prune
sudo docker system prune
sudo apt-get update
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker
```
2. 国内镜像源以及docker启动gpu的配置
   1. [Docker Hub 镜像加速器](https://gist.github.com/y0ngb1n/7e8f16af3242c7815e7ca2f0833d3ea6)
   2. `sudo vi /etc/docker/daemon.json `
   3. 重启docker `sudo systemctl restart docker`
```json
// 创建修改 /etc/docker/daemon.json 
{
    "registry-mirrors": [
        "https://dockerproxy.com",
        "https://docker.mirrors.ustc.edu.cn",
        "https://docker.nju.edu.cn"
    ],
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```
3. 镜像构建`Dockerfile`
```Dockerfile
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

RUN apt-get update
RUN apt-get install -y openmpi-bin 
RUN apt-get install -y libopenmpi-dev
RUN apt-get install -y ninja-build  
RUN apt-get install -y python3-mpi4py
RUN apt-get install -y numactl
RUN echo "Setting System Param >>>>>"
RUN echo "export PATH=/usr/bin/mpirun:\$PATH" >> ~/.bashrc
RUN echo "export PATH=/usr/bin/mpiexec:\$PATH" >> ~/.bashrc
RUN echo "export PATH=/opt/conda/bin/ninja:\$PATH" >> ~/.bashrc
RUN echo "export PATH=/usr/bin/mpirun:\$PATH" >> ~/.profile
RUN echo "export PATH=/usr/bin/mpiexec:\$PATH" >> ~/.profile
RUN echo "export PATH=/opt/conda/bin/ninja:\$PATH" >> ~/.profile

RUN echo 'export NUMA_POLICY=preferred' >> ~/.bashrc
RUN echo 'export NUMA_NODES=0' >> ~/.bashrc

RUN pip3 install deepspeed==0.12.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN conda install -c conda-forge mpi4py
RUN pip3 install tqdm   -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install triton  -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install tensorboard  -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN echo "Done"
```
4. 构建和查看镜像
```shell
sudo docker build -t ds_env_container . --network=host
sudo docker images
```

# 2 cifiar10 deepspeed训练代码解析



# 3 cifiar10 训练
## 3.1 本地训练
```shell
deepspeed --bind_cores_to_rank cifiar10_ds_train.py --deepspeed $@
```

## 3.2 Docker训练
```shell
# 启动docker
sudo docker run --env CUDA_VISIBLE_DEVICES=0 -it --ipc=host \
    --security-opt seccomp=seccomp.json \
    --gpus all \
    -v  /home/scc/sccWork/myGitHub/My_Learn/deepSpeed/ds_train_learning/cifiar:/app \
    -v  /home/scc/Downloads/Datas:/data \
    ds_env_container

# docker中运行
cd /app
deepspeed --bind_cores_to_rank cifiar10_ds_train.py --deepspeed $@

# 退出
exit
```
seccomp.json 是在遇到`set_mempolicy: Operation not permitted` 问题继续的解决
参考的 [https://github.com/bytedance/byteps/issues/17](https://github.com/bytedance/byteps/issues/17)








