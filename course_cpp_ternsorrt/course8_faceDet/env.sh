# 打包docker
sudo docker build -t trt_env . --network=host


# 本地 GPU
sudo apt-get update
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker

# 启动docker  OR sudo docker start facenet_env && sudo docker exec -it facenet_env /bin/bash  
sudo docker run -it --ipc=host \
    --gpus all -it  \
    --name facenet_env \
    -v `pwd`:/app  \
    trt_env

    # -p 1936:1935 -p 8556:8554 \



# Docker环境内
cd /app/facenet
# 下载检测模型
curl -LO 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/facenet/versions/pruned_quantized_v2.0.1/files/model.etlt'
curl -LO 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/facenet/versions/pruned_quantized_v2.0.1/files/int8_calibration.txt'

# download tao-converter
curl -LO 'https://api.ngc.nvidia.com/v2/resources/nvidia/tao/tao-converter/versions/v3.22.05_trt8.4_x86/files/tao-converter'

# 给运行权限
chmod +x ./tao-converter
#模型转换
./tao-converter -k nvidia_tlt -d 3,416,736 ./model.etlt -t int8 -c ./int8_calibration.txt


#编译facedet_test 并运行
cd ..
cmake -B build .
cmake --build build
./build/facedet_test --model facenet/saved.engine --img imges/test_face.jpg



