import onnx
onnx_model = onnx.load("x_vector.onnx")
for input in onnx_model.graph.input:
    print(input.name)