import importlib
import scipy
import speechbrain
import torch
import torchaudio
import onnx
import onnx_tf
import tensorflow
import tensorflow_addons
import numpy
import soundfile
import yaml
import hyperpyyaml
import tqdm
import joblib
import huggingface_hub
import sentencepiece
import onnxruntime
import tensorflow_probability

# 自定义函数获取版本，处理无 __version__ 的模块
def get_version(module):
    try:
        return module.__version__
    except AttributeError:
        return "Unknown (no __version__ attribute)"

print("验证包版本...")
print(f"scipy: {get_version(scipy)}")
print(f"speechbrain: {get_version(speechbrain)}")
print(f"torch: {get_version(torch)}")
print(f"torchaudio: {get_version(torchaudio)}")
print(f"onnx: {get_version(onnx)}")
print(f"onnx_tf: {get_version(onnx_tf)}")
print(f"tensorflow: {get_version(tensorflow)}")
print(f"tensorflow_addons: {get_version(tensorflow_addons)}")
print(f"numpy: {get_version(numpy)}")
print(f"soundfile: {get_version(soundfile)}")
print(f"pyyaml: {get_version(yaml)}")
print(f"hyperpyyaml: {get_version(hyperpyyaml)}")
print(f"tqdm: {get_version(tqdm)}")
print(f"joblib: {get_version(joblib)}")
print(f"huggingface_hub: {get_version(huggingface_hub)}")
print(f"sentencepiece: {get_version(sentencepiece)}")
print(f"onnxruntime: {get_version(onnxruntime)}")
print(f"tensorflow_probability: {get_version(tensorflow_probability)}")
print("GPU Available:", torch.cuda.is_available())  # 应为 False