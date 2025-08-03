# check_wav.py
import torchaudio
import os
sample_path = "D:/speechbrain_project/sample.wav"
if not os.path.exists(sample_path):
    print(f"文件 {sample_path} 不存在！")
    exit(1)
try:
    signal, fs = torchaudio.load(sample_path)
    print(f"采样率: {fs}, 通道数: {signal.shape[0]}, 时长: {signal.shape[1]/fs:.2f}秒")
except Exception as e:
    print(f"加载音频失败: {e}")