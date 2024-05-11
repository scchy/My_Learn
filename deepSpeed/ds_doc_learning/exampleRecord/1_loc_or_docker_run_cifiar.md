# deepspeed 单机单卡本地运行 & Docker运行

本文笔者基于官方示例`DeepSpeedExamples/training/cifar/cifar10_deepspeed.py`进行本地构建和Docker构建运行示例（<font color="darkred">下列代码中均是踩坑后可执行的代码，尤其是Docker部分</font>）, 全部code可以看[笔者的github: cifiar10_ds_train.py](https://github.com/scchy/My_Learn/tree/master/deepSpeed/ds_train_learning/cifiar/cifiar10_ds_train.py)


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

res="""
REPOSITORY                TAG       IMAGE ID       CREATED       SIZE
ds_env_container          latest    9149b06c79c8   1 hours ago   17.4GB
ultralytics/ultralytics   latest    9d605fba39ec   6 weeks ago   13.8GB
"""
```

# 2 cifiar10 deepspeed训练代码解析

## 2.1 训练数据 & 简单CNN

### Data

直接`torchvision.datasets`中下载
```python
tf_func = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load or download cifar data
tr_set = torchvision.datasets.CIFAR10(
    root=DATA_FILE_NOW,
    train=True,
    download=True, 
    transform=tf_func
)
testset = torchvision.datasets.CIFAR10(
    root=DATA_FILE_NOW, 
    train=False, 
    download=True, 
    transform=tf_func
)
```

### CNN
简单结构：CNN -> MLP -> head
其中包含[`Mixture of Experts`（专家混和架构）](https://deepspeed.readthedocs.io/en/stable/moe.html), [Mixture-of-Experts (MoE) 经典论文一览](https://zhuanlan.zhihu.com/p/542465517)
MOE模型的关键特点包括：
- 专业化：每个专家被训练来处理输入数据的不同部分或不同类型的任务。
- 灵活性：MOE模型可以根据任务需求动态地调整专家的数量和类型。
- 扩展性：通过增加更多的专家，MOE模型可以扩展以处理更复杂的任务。
- 并行处理：专家可以并行工作，这提高了模型的计算效率。
- 负载均衡：MOE模型可以通过将任务分配给多个专家来实现负载均衡。

```python
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.extra_feat = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.hiddens = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.head = nn.Linear(84, 10)
        self.moe_layer_list = []
        self.moe = cfg.moe
        if self.moe:
            expert=nn.Linear(84, 84)
            for n_e in cfg.num_experts:
                self.moe_layer_list.append(
                    deepspeed.moe.layer.MoE(
                        hidden_size=84,
                        expert=expert, 
                        num_expert=n_e,
                        ep_size=cfg.ep_word_size,
                        use_residual=cfg.mlp_type == 'residual',
                        k=cfg.top_k,
                        min_capacity=cfg.min_capacity,
                        nosiy_gate_policy=cfg.nosiy_gate_policy,
                    )
                )
            self.moe_layer_list = nn.ModuleList(self.moe_layer_list)

    def forward(self, x):
        x = self.extra_feat(x)
        x = self.hiddens(x)
        if self.moe:
            for layer in self.moe_layer_list:
                x, _, _ = layer(x)
        return self.head(x)
```

## 2.2 设置分布式环境 & 初始化deepSpeed训练模型

### 设置分布式环境

`deepspeed.init_distributed()` 
- 其中用`torch.distributed.get_rank()` 获取当前进程在所有分布式进程中的索引
  - 当索引不为0时，调用`torch.distributed.barrier()`使得进程进行等待
  - 当索引为0时，下载数据后，再调用`torch.distributed.barrier()`
    - 这时候所有进程都达到`barrier`, 然后进程各自进行后续操作

```python
if torch.distributed.get_rank() != 0:
    torch.distributed.barrier()

....

if torch.distributed.get_rank() == 0:
    torch.distributed.barrier()
```

###  初始化deepSpeed训练模型

```python
model_engine, opt, tr_loader, _lr_scheduler = deepspeed.initialize(
    args=cfg,  # 其中包含 local_rank 和 deepspeed_config 字段。如果传递了 config，则此对象为可选项。
    model=net, # torch.nn.Module
    model_parameters=params_grad, # 开启梯度下降的参数
    training_data=tr_set, # torch.utils.data.Dataset
    config=ds_config #  Instead of requiring args.deepspeed_config
)
```

config 设置的全部参数可以看 [www.deepspeed.ai 官方文档](https://www.deepspeed.ai/docs/config-json/)
这里我们仅仅对部分必要参数进行了简单设置。
其中ZeR（Zero Redundancy Optimizer）:
1. ZeRO的主要思想包括：
   - **数据并行主义的内存优化**：在传统的数据并行训练中，每个GPU都会存储一份完整的模型副本，这会导致显著的内存浪费。ZeRO通过将模型的参数、梯度和优化器状态分散到多个GPU上，从而减少了每个GPU所需的内存量。
   - **计算和通信的重叠**：ZeRO利用异步执行和管道化技术，使得计算和通信可以并行进行，这样可以进一步提高训练的效率。
   - **动态损失缩放**：为了混合精度训练的稳定性，ZeRO会自动调整损失缩放因子，避免了由于数值不稳定导致的问题。
   - **优化器状态的CPU offloading**：ZeRO允许将优化器的状态存储在CPU上，进一步减少了GPU的内存占用。

2. ZeRO分为几个阶段（stages），每个阶段都提供了不同程度的优化：
   - ZeRO-Offload：将优化器的状态和计算卸载到CPU。
   - ZeRO-Stage 1：在数据并行的基础上，对模型参数进行划分，每个GPU只存储一部分模型参数。
   - ZeRO-Stage 2：除了参数划分，还将梯度进行划分，每个GPU只存储与其参数相对应的梯度。
   - ZeRO-Stage 3：进一步将模型的前向和后向计算分散到不同的GPU上。

```python
ds_config = {
    "train_batch_size": 16, # = train_micro_batch_size_per_gpu * gradient_accumulation_steps * number of GPUs
    "train_micro_batch_size_per_gpu": 16,
    "gradient_accumulation_steps": 1, # 梯度累积
    "steps_per_print": 2000, # logging相关 每 N 个培训步骤打印进度报告
    "tensorboard": { # tensorboard 相关设置
        "enabled": True,
        "output_path": "deepspeed_runs",
        "job_name": "cifiar10-try"
    },
    "optimizer":{ # 优化器相关设置
        'type': 'Adam',
        'params': {
            "lr": 0.001,
            "betas": [0.8, 0.999],
            'eps': 1e-8,
            'weight_decay': 3e-7
        }
    },
    "scheduler": { # 学习率相关设置
        'type': 'WarmupLR',
        'params': {
            'warmup_min_lr': 0,
            'warmup_max_lr': 0.001,
            'warmup_num_steps': 1000
        }
    },
    "gradient_clipping": 1.0, # 梯度裁剪
    'prescale_gradients': False, # 在进行 allreduce 之前缩放梯度
    'bf16': {'enabled': cfg.dtype == 'bf16'},
    "fp16": {
                "enabled": cfg.dtype == "fp16",
                "fp16_master_weights_and_grads": False,
                "loss_scale": 0, # the loss scaling value for FP16 training
                "loss_scale_window": 500,
                # "hysteresis": 2,
                "min_loss_scale": 1, 
                "initial_scale_power": 15, # 2^initial_scale_power the power of the initial dynamic loss scale valu
            },
    "wall_clock_breakdown": False, # 为  forward/backward/update 训练阶段的延迟计时
    "zero_optimization": { # ZeRO（Zero Redundancy Optimizer） memory optimizations
        "stage": cfg.stage,
        "allgather_partitions": True, # 以便在每一步结束时从所有 GPU 收集更新参数
        "reduce_scatter": True, # 使用 reduce 或 reduce scatter 而不是 allreduce 来平均梯度
        "allgather_bucket_size": 5e8, # 一次全采集的元素数量。限制大尺寸模型全收集所需的内存
        "reduce_bucket_size": 5e8, # 一次还原/全还原的元素数量。限制大型模型的 allgather 内存需求
        "overlap_comm": True, # 尝试将梯度缩减与逆向计算相重叠
        "contiguous_gradients": True, # 在生成梯度时将其复制到连续的缓冲区中。避免在后向传递过程中出现内存碎片
        "cpu_offload": False # 将优化器内存和计算卸载到 CPU
    }
}

```


# 3 cifiar10 训练

再2中准备了`model_engine, opt, tr_loader, _lr_scheduler `, 就可以按照一般的深度学习方式进行训练了
```python
from deepspeed.accelerator import get_accelerator

# 获取分布的机器的device
local_rank = model_engine.local_rank
local_device = get_accelerator().device_name(local_rank)
for ep in range(cfg.epochs):
    for i, data_b in enumerate(tr_loader):
        ipts, labels = data_b[0].to(local_device), data_b[1].to(local_device)

        out = model_engine(ipts)
        loss = criterion(out, labels)
        # 一般DL loss.backward()
        model_engine.backward(loss)
        # 一般DL opt.step()
        model_engine.step()

```

## 3.1 本地训练
```shell
deepspeed --bind_cores_to_rank cifiar10_ds_train.py --deepspeed $@

res="""
[ 030 / 30 ]: 100%|█████████████████████████████████████| 30/30 [02:29<00:00,  4.97s/it, loss=0.504]
Finished Training
Accuracy of the network on the 10000 test images:  57 %
Accuracy of plane :  67 %
Accuracy of   car :  70 %
Accuracy of  bird :  43 %
Accuracy of   cat :  41 %
Accuracy of  deer :  51 %
Accuracy of   dog :  42 %
Accuracy of  frog :  66 %
Accuracy of horse :  61 %
Accuracy of  ship :  66 %
Accuracy of truck :  59 %
"""
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








