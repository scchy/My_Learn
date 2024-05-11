# python3
# Create Date: 20240504
# Author: Scc_hy
# Function: train cifiar10 classification with deepspeed
# Tip: docker
# RUN:  deepspeed --bind_cores_to_rank cifiar10_ds_train.py --deepspeed $@
# ===================================================================================

from argparse import Namespace
import deepspeed 
import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision 
from torchvision import transforms
from deepspeed.accelerator import get_accelerator
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
from tqdm.auto import tqdm

LOC_RUN_FILE = '/home/scc/Downloads/Datas'
DOCKER_RUN_FILE = '/data'
DATA_FILE_NOW = DOCKER_RUN_FILE


config = Namespace(
    epochs=30,
    local_rank=-1,
    log_interval=2000,
    dtype='fp16',
    stage=0,
    moe=False,
    ep_word_size=1,
    num_experts=[1,],
    mlp_type='standard', # redsiduel
    top_k=1,
    min_capacity=0,
    nosiy_gate_policy=None,
    moe_param_group=False
)

def create_moe_param_groups(model):
    """Create separate parameter groups for each expert."""
    parameters = {"params": [p for p in model.parameters()], "name": "parameters"}
    return split_params_into_different_moe_groups_for_optimizer(parameters)


def get_ds_config(cfg):
    # https://www.deepspeed.ai/docs/config-json/
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
                    "hysteresis": 2, # the delay shift in dynamic loss scaling.
                    "min_loss_scale": 1, # the minimum dynamic loss scale value.
                    "initial_scale_power": 15, # 2^initial_scale_power the power of the initial dynamic loss scale valu
                },
        "wall_clock_breakdown": False, # 为  forward/backward/update 训练阶段的延迟计时
        # 
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
    return ds_config


class MLP(nn.Module):
    """
    [`Mixture of Experts`（专家混和架构）](https://deepspeed.readthedocs.io/en/stable/moe.html)
    [Mixture-of-Experts (MoE) 经典论文一览](https://zhuanlan.zhihu.com/p/542465517)
    """
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
                        expert=expert,  # 专家模型
                        num_expert=n_e, # default=1, 每层专家总数
                        ep_size=cfg.ep_word_size, #  default=1,
                        use_residual=cfg.mlp_type == 'residual', 
                        k=cfg.top_k, # default=1,  only supports k=1 or k=2. 
                        min_capacity=cfg.min_capacity, # default=4, 每个专家最小容量
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

        
def test(model_engine, testset, local_device, target_dtype, test_batch_size=4):
    """_summary_

    Args:
        model_engine (deepspeed.runtime.engine.DeepSpeedEngine): the DeepSpeed engine.
        testset (torch.utils.data.Dataset): test dataset
        local_device (str): the local device name
        target_dtype (torch.type): the target datatype for the test data
        test_batch_size (int, optional): the test batch size. Defaults to 4.
    """
    # The 10 classes for CIFAR10.
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    # Define the test dataloader
    testloader = DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=0
    )
    # For total accuracy
    correct, total = 0, 0
    # For accuracy per class
    class_correct = [0.0 for _ in range(10)]
    class_total = [0.0 for _ in range(10)]
    # Start testing
    model_engine.eval()
    with torch.no_grad():
        for img, labels in testloader:
            if target_dtype != None:
                img = img.to(target_dtype)
            out = model_engine(img.to(local_device))
            _, predict = torch.max(out.data, dim=1)
            total += labels.size(0)
            correct += (predict == labels.to(local_device)).sum().item()
            # Cout the accuracy per class
            batch_correct =  (predict == labels.to(local_device)).squeeze()
            for i in range(test_batch_size):
                label = labels[i]
                class_correct[label] += batch_correct[i].item()
                class_total[label] += 1

    if model_engine.local_rank == 0:
        print(
            f"Accuracy of the network on the {total} test images: {100 * correct / total : .0f} %"
        )

        # For all classes, print the accuracy.
        for i in range(10):
            print(
                f"Accuracy of {classes[i] : >5s} : {100 * class_correct[i] / class_total[i] : 2.0f} %"
            )


def main(cfg):
    # Initialize DeepSpeed distributed backend 
    print( "Start Init_distributed", ">>"*20)
    deepspeed.init_distributed()
    print( "Finished Init_distributed", ">>"*20)
    # Step1 Data Prepare
    tf_func = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if torch.distributed.get_rank() != 0:
        # Might be downloading cifar data, let rank 0 download first
        torch.distributed.barrier()
    
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
    if torch.distributed.get_rank() == 0:
        # Cifar data is downloaded, indicate other ranks can proceed.
        torch.distributed.barrier()
    
    # Step2: Defined net
    net = MLP(cfg)
    # Get list of parameters that require gradients
    params_grad = filter(lambda p: p.requires_grad, net.parameters())
    # if using MoE:
    if cfg.moe_param_group:
        params_grad = create_moe_param_groups(net)
    
    # Initialize DeepSpeed to use the following features
    # 1) Distributed model.
    # 2) Distributed data loader
    # 3) DeepSpeed optimizer
    ds_config = get_ds_config(cfg)
    model_engine, opt, tr_loader, _lr_scheduler = deepspeed.initialize(
        args=cfg,  # 其中包含 local_rank 和 deepspeed_config 字段。如果传递了 config，则此对象为可选项。
        model=net, # torch.nn.Module
        model_parameters=params_grad, # 开启梯度下降的参数
        training_data=tr_set, # torch.utils.data.Dataset
        config=ds_config #  Instead of requiring args.deepspeed_config
    )

    # Get the local device name 
    local_rank = model_engine.local_rank
    local_device = get_accelerator().device_name(local_rank)
    # For float32, target_type will be None so no datatype conversition
    target_dtype = None
    if model_engine.bfloat16_enabled():
        target_dtype = torch.bfloat16
    elif model_engine.fp16_enabled():
        target_dtype = torch.half

    # Define the classification Cross-Enrtopy loss function
    criterion = nn.CrossEntropyLoss()

    # Step3: traning
    ep_bar = tqdm(range(cfg.epochs))
    for ep in ep_bar:
        running_loss = 0.0
        ep_bar.set_description(f'[ {str(ep+1).zfill(3)} / {cfg.epochs} ]')
        for i, data_b in enumerate(tr_loader):
            ipts, labels = data_b[0].to(local_device), data_b[1].to(local_device)
            if target_dtype != None:
                ipts = ipts.to(target_dtype)
            
            out = model_engine(ipts)
            loss = criterion(out, labels)
            model_engine.backward(loss)
            model_engine.step()
            # print statistics
            running_loss += loss.item()
            if local_rank == 0 and i % cfg.log_interval == (
                cfg.log_interval - 1
            ):  # Print every log_interval mini-batches.
                ep_bar.set_postfix({
                    "loss": f'{running_loss / cfg.log_interval : .3f}'
                })
                print(
                    f"[{ep + 1 : d}, {i + 1 : 5d}] loss: {running_loss / cfg.log_interval : .3f}"
                )
                running_loss = 0.0
    print("Finished Training")

    # Step 4. Test the network on the test data.
    test(model_engine, testset, local_device, target_dtype)


if __name__ == "__main__":
    main(config)
    res = """
    Accuracy of the network on the 10000 test images:  58 %
    Accuracy of plane :  61 %
    Accuracy of   car :  74 %
    Accuracy of  bird :  46 %
    Accuracy of   cat :  43 %
    Accuracy of  deer :  52 %
    Accuracy of   dog :  46 %
    Accuracy of  frog :  55 %
    Accuracy of horse :  65 %
    Accuracy of  ship :  73 %
    Accuracy of truck :  62 %
    [2024-05-11 10:53:25,019] [INFO] [launch.py:351:main] Process 1821942 exits successfully.
    """

