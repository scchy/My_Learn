

yolo detect train \
    data=./visDrone2019.yaml \
    model=/home/scc/Downloads/AIToy/P2_Yolov8CarCount/3.train_own_data/weights/yolov8n.pt \
    epochs=100 imgsz=800 workers=4 batch=16 \
    project=yoloTrain__VisDrone2019 \
    name=n_100_800

