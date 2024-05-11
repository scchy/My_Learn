#  Docker Hub 镜像加速器 https://gist.github.com/y0ngb1n/7e8f16af3242c7815e7ca2f0833d3ea6
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
# 创建修改 /etc/docker/daemon.json 
# {
#     "registry-mirrors": [
#         "https://dockerproxy.com",
#         "https://docker.mirrors.ustc.edu.cn",
#         "https://docker.nju.edu.cn"
#     ],
#     "default-runtime": "nvidia",
#   "runtimes": {
#     "nvidia": {
#       "path": "nvidia-container-runtime",
#       "runtimeArgs": []
#     }
#   }
# }

# clear cache
sudo docker builder prune
sudo docker system prune

# delete 
sudo docker ps -a
sudo docker rm -f ec1bf0ee73c9
# delete image
sudo docker rmi ds_env_container

# GPU
sudo apt-get update
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker

# build image
sudo docker build -t ds_env_container . --network=host
sudo docker images

# RUN
sudo docker run --env CUDA_VISIBLE_DEVICES=0 -it --ipc=host \
    --security-opt seccomp=seccomp.json \
    --gpus all \
    -v  /home/scc/sccWork/myGitHub/My_Learn/deepSpeed/ds_train_learning/cifiar:/app \
    -v  /home/scc/Downloads/Datas:/data \
    ds_env_container

# inner-docker:  numactl --hardware
# inner-docker:  deepspeed --bind_cores_to_rank cifiar10_ds_train.py --deepspeed $@
# ERROR: set_mempolicy: Operation not permitted
#      SOLVE: https://github.com/bytedance/byteps/issues/17
# inner-docker:  exit


# STOP
systemctl stop docker
 