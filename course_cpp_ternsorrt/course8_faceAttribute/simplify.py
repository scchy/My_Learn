from torch import onnx 
import onnx 
import sys 
import onnxsim

model_onnx = onnx.load(sys.argv[1])
onnx.checker.check_model(model_onnx)

model_onnx, check = onnxsim.simplify(model_onnx)
output_file_name = sys.argv[1].split(".")[0] + "_sim.onnx"
onnx.save(model_onnx, output_file_name)
