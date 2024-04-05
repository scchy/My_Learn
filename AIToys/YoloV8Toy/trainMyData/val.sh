
model_path=yoloTrain__VisDrone2019/n_100_8002/weights/best.pt
echo "======================================================================="
yolo detect val data=./visDrone2019.yaml  model=${model_path}


echo "======================================================================="
test_jpg=/home/scc/Downloads/AIToy/P2_Yolov8CarCount/4.verify_train_result/test.jpg
test_mp4=/home/scc/Downloads/AIToy/P2_Yolov8CarCount/4.verify_train_result/test.mp4
model_path=yoloTrain__VisDrone2019/n_100_8002/weights/best.pt

yolo predict  model=${model_path} source=${test_jpg}
