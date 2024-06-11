#  OR sudo docker start env_pyt_1.12  
#    && sudo docker exec -it env_pyt_1.12  /bin/bash  
sudo docker run --gpus all -it --name env_pyt_1.12 \
    -v $(pwd):/app \
    -v "/home/scc/Downloads/TRT/7.facenet人脸检测识别属性/6.attributes_export":/data \
    --network=host \
    nvcr.io/nvidia/pytorch:22.03-py3 

# ***************************************************************************************************************\
# Export
cd /app
pip install tensorflow  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tf2onnx -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install onnxsim -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
# 性别
python -m tf2onnx.convert --saved-model /data/model/model_gender  --output /app/export_onnx/gender.onnx --opset 10
# 年龄
python -m tf2onnx.convert --saved-model /data/model/model_age  --output /app/export_onnx/age.onnx --opset 10
# 口罩
python -m tf2onnx.convert --saved-model /data/model/model_mask --output /app/export_onnx/mask.onnx --opset 10
# 表情
python -m tf2onnx.convert --saved-model /data/model/model_emotion  --output /app/export_onnx/emotion.onnx --opset 10

# 简化
python simplify.py /app/export_onnx/gender.onnx
python simplify.py /app/export_onnx/age.onnx
python simplify.py /app/export_onnx/mask.onnx
python simplify.py /app/export_onnx/emotion.onnx

# **************************************************************************************************************
# Test
sudo docker run -it --ipc=host \
    --gpus all -it  \
    --name attribute_proj \
    -v `pwd`:/app \
    trt_env

cd /app/trt_proj

cmake -B build .
cmake --build build

# 转TRT engine（以表情分类模型为例）
./build/build --onnx_file /app/export_onnx/emotion_sim.onnx \
    --input_h 48 --input_w 48 --input_c 1 --format nhwc

./build/build --onnx_file /app/export_onnx/gender_sim.onnx \
    --input_h 48 --input_w 48 --input_c 1 --format nhwc

./build/build --onnx_file /app/export_onnx/age_sim.onnx \
    --input_h 48 --input_w 48 --input_c 1 --format nhwc

./build/build --onnx_file /app/export_onnx/mask_sim.onnx \
    --input_h 48 --input_w 48 --input_c 1 --format nhwc

# 性别测试
./build/attribute_test --model /app/export_onnx/gender_sim.engine --type gender --img ./images/1.gender/man.png

./build/attribute_test --model /app/export_onnx/gender_sim.engine --type gender --img ./images/1.gender/woman.png

# 年龄测试
./build/attribute_test --model /app/export_onnx/age_sim.engine --type age --img ./images/2.age/old.png

./build/attribute_test --model /app/export_onnx/age_sim.engine --type age --img ./images/2.age/young.png

# 口罩测试
./build/attribute_test --model /app/export_onnx/mask_sim.engine --type mask --img ./images/3.mask/unmask.jpg

./build/attribute_test --model /app/export_onnx/mask_sim.engine --type mask --img ./images/3.mask/mask.png

# 表情测试
./build/attribute_test --model /app/export_onnx/emotion_sim.engine --type emotion --img ./images/4.emotion/angry.jpg

./build/attribute_test --model /app/export_onnx/emotion_sim.engine --type emotion --img ./images/4.emotion/sad.jpg

# **************************************************************************************************************
# App


