# test_xvector.py
import speechbrain as sb
import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition
import os

print("验证环境...")
try:
    print(f"speechbrain: {sb.__version__}")
    print(f"torch: {torch.__version__}")
    print(f"torchaudio: {torchaudio.__version__}")
except ImportError as e:
    print(f"导入错误: {e}")
    exit(1)

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

try:
    signal, fs = torchaudio.load(sample_path)
except Exception as e:
    print(f"加载音频失败: {e}")
    exit(1)

if fs != 16000:
    print(f"警告：采样率 {fs} 不为 16kHz，正在重采样...")
    resampler = torchaudio.transforms.Resample(fs, 16000)
    signal = resampler(signal)
if signal.shape[0] > 1:
    print("警告：检测到多声道音频，正在转换为单声道...")
    signal = torch.mean(signal, dim=0, keepdim=True)
if signal.shape[1] / fs < 5:
    print("警告：音频时长小于 5 秒，建议使用 5-10 秒音频以获得更好结果")

embedding = model.encode_batch(signal)
print(f"原始嵌入形状: {embedding.shape}")
embedding = embedding.squeeze(1)  # 仅移除时间维度
print(f"调整后嵌入向量形状: {embedding.shape}")  # 预期：torch.Size([1, 512])