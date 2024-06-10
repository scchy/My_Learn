# python export.py --weights="/home/scc/Downloads/TRT/7.facenet人脸检测识别属性/3.facenet_export/weights/facenet_best.pt"

from FACENET import InceptionResnetV1
from torchsummary import summary
import torch 
import argparse

parser = argparse.ArgumentParser(description='Facenet export')
parser.add_argument('--weights', type=str, default='weights/facenet_best.pt', help='model name')
args = parser.parse_args()

weights_file = args.weights
facenet = InceptionResnetV1(
            is_train=False, embedding_length=128, num_classes=14575)
device = 'cpu'

facenet.to(device)
facenet.load_state_dict(torch.load(weights_file, map_location=device))
facenet.eval()

# summary(facenet, (3, 112, 112))
x = torch.randn(1, 3, 112, 112, requires_grad=False)
# Export 
torch.onnx.export(
    facenet,
    x, 
    "facenet.onnx",
    export_params=True, 
    opset_version=10,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0:'batch_size'}, 'output': {0: "batch_size"}}
)
print("Export of facenet.onnx complete!")


# SIMPLIFY
import onnx 
import onnxsim 

model_onnx = onnx.load("facenet.onnx")
onnx.checker.check_model(model_onnx)
model_onnx, check = onnxsim.simplify(model_onnx)
onnx.save(model_onnx, 'facenet_sim.onnx')
print('Export of facenet_sim.onnx complete !')


