# **************************************************************************************************************
# App
cd ~/sccWork/myGitHub/My_Learn/course_cpp_ternsorrt/course8_nanoApp

# OR sudo docker start app_proj
#    &&     sudo docker exec -it app_proj /bin/bash  
sudo docker run -it \
    -p 1936:1935  -p 8556:8554 \
    --gpus all -it  \
    --name app_proj \
    -v `pwd`:/app \
    --device=/dev/video0:/dev/video0 \
    trt_env

# -------------------------------------------------------
# step1 download facedet model
cd /app/TAO 
curl -LO 'https://api.ngc.nvidia.com/v2/resources/nvidia/tao/tao-converter/versions/v3.22.05_trt8.4_aarch64/files/tao-converter'
chmod +x tao-converter
curl -LO 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/facenet/versions/pruned_quantized_v2.0.1/files/model.etlt'
curl -LO 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/facenet/versions/pruned_quantized_v2.0.1/files/int8_calibration.txt'


./tao-converter -k nvidia_tlt -d 3,416,736 model.etlt \
    -t int8 -c int8_calibration.txt  -e detect.engine
# ---------------------------------------------------------
# step 2 build 
cd /app
cmake -B build .
cmake --build build

# ------------------------------------------------------------
# step 3 export model 
cd /app/backup_onnx/facenet_export
python export.py --weights="/data/3.facenet_export/weights/facenet_best.pt"

cd /app
pip install tensorflow  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tf2onnx -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install onnxsim -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
# 性别
python -m tf2onnx.convert --saved-model /data/model/model_gender  --output /app/backup_onnx/gender.onnx --opset 10
# 年龄
python -m tf2onnx.convert --saved-model /data/model/model_age  --output /app/backup_onnx/age.onnx --opset 10
# 口罩
python -m tf2onnx.convert --saved-model /data/model/model_mask --output /app/backup_onnx/mask.onnx --opset 10
# 表情
python -m tf2onnx.convert --saved-model /data/model/model_emotion  --output /app/backup_onnx/emotion.onnx --opset 10

# 简化
python simplify.py /app/backup_onnx/gender.onnx
python simplify.py /app/backup_onnx/age.onnx
python simplify.py /app/backup_onnx/mask.onnx
python simplify.py /app/backup_onnx/emotion.onnx


# ------------------------------------------------------------
# step 4 build model 
# facenet识别模型
./build/build -onnx_file /app/backup_onnx/facenet_sim.onnx --input_h 112 --input_w 112 

# 属性模型
./build/build --onnx_file /app/backup_onnx/gender_sim.onnx --input_h 48 --input_w 48 --input_c 1 --format nhwc

./build/build --onnx_file /app/backup_onnx/age_sim.onnx --input_h 48 --input_w 48 --input_c 1 --format nhwc

./build/build --onnx_file /app/backup_onnx/emotion_sim.onnx --input_h 48 --input_w 48 --input_c 1 --format nhwc

./build/build --onnx_file /app/backup_onnx/mask_sim.onnx --input_h 48 --input_w 48 --input_c 1 --format nhwc


# ------------------------------------------------------------
# 构建
export PATH=$PATH:/usr/local/cuda/bin

# install ffmpeg
apt install ffmpeg

# host ************************* VLC
# reference: https://zhuanlan.zhihu.com/p/669695580?utm_id=0
# apt install vlc
sudo apt-get install flatpak
sudo apt-get install gnome
sudo apt install gnome-software-plugin-flatpak
sudo  flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
sudo reboot
flatpak install vlc
# host *************************

# 测试
./build/stream \
--facedet ./TAO/detect.engine \
--facenet ./backup_onnx/facenet_sim.engine \
--att_gender ./backup_onnx/gender_sim.engine \
--att_age ./backup_onnx/age_sim.engine \
--att_emotion ./backup_onnx/emotion_sim.engine \
--att_mask ./backup_onnx/mask_sim.engine \
--faces ./face_list.txt \
--vid  rtsp://localhost:8554/camera

# --vid rtsp://localhost:8554/live1.sdp
# --vid ./rtmp_server/test.mp4

# host VLC
#   rtsp://localhost:8555/live1.sdp
#   rtmp://localhost:1936/live

# change 
ffmpeg -i qq.mp4 -c:v libx264 -profile:v high -crf 22 -preset slow qq_new.mp4
# 使用 FFMPEG 来旋转视频并保存新的视频文件: 这里的 transpose=1 参数将视频顺时针旋转90度
ffmpeg -i QQ_202411.mp4 -vf "transpose=1" QQ_202411_new.mp4


./build/stream \
--facedet ./TAO/detect.engine \
--facenet ./backup_onnx/facenet_sim.engine \
--att_gender ./backup_onnx/gender_sim.engine \
--att_age ./backup_onnx/age_sim.engine \
--att_emotion ./backup_onnx/emotion_sim.engine \
--att_mask ./backup_onnx/mask_sim.engine \
--faces ./face_list.txt \
--vid ./rtmp_server/QQ_202411_new.mp4

