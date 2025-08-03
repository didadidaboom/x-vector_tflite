# convert_xvector.py
import speechbrain as sb
import torch
import torchaudio
import tensorflow as tf
import onnx
import onnx_tf
import onnxruntime
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.lobes.features import Fbank
import numpy as np
import os
import soundfile

print("验证包版本...")
print(f"speechbrain: {sb.__version__}")
print(f"torch: {torch.__version__}")
print(f"torchaudio: {torchaudio.__version__}")
print(f"onnx: {onnx.__version__}")
print(f"onnx_tf: {onnx_tf.__version__}")
print(f"tensorflow: {tf.__version__}")
print(f"numpy: {np.__version__}")
print(f"soundfile: {soundfile.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")

print("加载 x-vector 模型...")
model = SpeakerRecognition.from_hparams(
    source="D:/speechbrain_project/pretrained_models/spkrec-xvect-voxceleb",
    savedir="D:/speechbrain_project/pretrained_models/spkrec-xvect-voxceleb"
)

print("请提供一个 16kHz 单声道 WAV 文件（5-10 秒）。")
sample_path = input("输入 sample.wav 的路径（例如 D:\\speechbrain_project\\sample.wav）：").strip('"')
sample_path = sample_path.replace('\\', '/').strip()
if not os.path.exists(sample_path):
    print(f"文件 {sample_path} 不存在！")
    exit(1)

signal, fs = torchaudio.load(sample_path)
if fs != 16000:
    print("重采样到 16kHz...")
    resampler = torchaudio.transforms.Resample(fs, 16000)
    signal = resampler(signal)
if signal.shape[0] > 1:
    print("转换为单声道...")
    signal = torch.mean(signal, dim=0, keepdim=True)
if signal.shape[1] / fs < 5:
    print("警告：音频时长小于 5 秒，建议使用 5-10 秒音频")

print("提取 MFCC...")
fbank = Fbank(n_mels=24)  # 通道数为 24，与模型兼容
mfcc = fbank(signal)
print(f"MFCC 形状: {mfcc.shape}")

print("导出模型到 ONNX...")
class XVectorWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.mods.embedding_model
        self.norm = model.mods.mean_var_norm_emb
    def forward(self, x):
        x = self.model(x)
        lengths = torch.ones(x.shape[0], device=x.device)
        x = self.norm(x, lengths)
        return x.squeeze(1)

model_wrapper = XVectorWrapper(model)
model_wrapper.eval()
dummy_input = torch.randn(1, 1089, 24)  # 匹配实际 MFCC 形状
torch.onnx.export(
    model_wrapper,
    dummy_input,
    "x_vector.onnx",
    input_names=["mfcc"],
    output_names=["embedding"],
    dynamic_axes={
        "mfcc": {0: "batch", 1: "time"},
        "embedding": {0: "batch"}
    },
    opset_version=11,
    verbose=False
)

print("验证 ONNX 模型...")
onnx_model = onnx.load("x_vector.onnx")
onnx.checker.check_model(onnx_model)
onnx_session = onnxruntime.InferenceSession("x_vector.onnx", providers=['CPUExecutionProvider'])
onnx_input = {"mfcc": mfcc.numpy()}
onnx_output = onnx_session.run(None, onnx_input)[0]
print(f"ONNX 输出形状: {onnx_output.shape}")

print("将 ONNX 转换为 TensorFlow...")
tf_model = onnx_tf.backend.prepare(onnx_model)
tf_model.export_graph("x_vector_pb")

print("转换为 TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model("x_vector_pb")
converter.optimizations = []  # 禁用优化
converter.experimental_select_tf_ops = True
converter.allow_custom_ops = True
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
converter.experimental_new_converter = True  # 使用 MLIR 转换器
converter.experimental_enable_dynamic_shapes = True  # 明确启用动态形状
tflite_model = converter.convert()
with open("x_vector.tflite", "wb") as f:
    f.write(tflite_model)

print("验证 TFLite 模型...")
interpreter = tf.lite.Interpreter(model_path="x_vector.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite 输入详情:", input_details)
print("TFLite 输出详情:", output_details)
# 动态调整输入形状以匹配 MFCC
interpreter.resize_tensor_input(input_details[0]["index"], [1, mfcc.shape[1], 24])
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]["index"], mfcc.numpy())
interpreter.invoke()
tflite_output = interpreter.get_tensor(output_details[0]["index"])
print(f"TFLite 输出形状: {tflite_output.shape}")