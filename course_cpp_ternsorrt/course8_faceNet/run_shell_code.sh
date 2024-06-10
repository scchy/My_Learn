
# **********************************************************************************
# train & export
sudo docker pull nvcr.io/nvidia/pytorch:22.03-py3 

#  OR sudo docker start env_pyt_1.12  
#    && sudo docker exec -it env_pyt_1.12  /bin/bash  
sudo docker run --gpus all -it --name env_pyt_1.12 \
    -v $(pwd):/app \
    -v "/home/scc/Downloads/TRT/7.facenet人脸检测识别属性/":/data \
    nvcr.io/nvidia/pytorch:22.03-py3 

# export 
cd /app/facenet_export 
python export.py --weights="/data/3.facenet_export/weights/facenet_best.pt"


# 生成TensorRT engine
cd /app/facenet_trt
./build/build -onnx_file /app/facenet_export/facenet_sim.onnx --input_h 112 --input_w 112 

# ************************************************************************************
#  OR sudo docker start facenet
#    && sudo docker exec -it facenet  /bin/bash  
sudo docker run -it --ipc=host \
    --gpus all -it  \
    --name facenet \
    -v `pwd`:/app  \
    -v "/home/scc/Downloads/TRT/7.facenet人脸检测识别属性/":/data \
    trt_env


# Docker环境内
cd /app/facenet
# use direct trt env convert model
# 给运行权限
chmod +x ./tao-converter
#模型转换
./tao-converter -k nvidia_tlt -d 3,416,736 ./model.etlt -t int8 -c ./int8_calibration.txt -e detect.engine



cd /app/facenet_trt
cmake -B build .
cmake --build build

./build/build -onnx_file /app/factnet_export/facenet_sim.onnx --input_h 112 --input_w 112 

./build/facenet_test --facedet=/app/facenet/detect.engine --facenet=/app/factnet_export/facenet_sim.engine \
    --img ./test1.jpg  --faces=/data/4.facenet/face_list_docker.txt

# ************************************************************************************


