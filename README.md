# 将 SpeechBrain x-vector 转换为 TFLite 以集成到 Flutter / Converting SpeechBrain x-vector to TFLite for Flutter Integration

[English](#english-version) | [中文](#中文版本)

## 中文版本

### 项目概述
本项目提供将 SpeechBrain x-vector 模型（`spkrec-xvect-voxceleb`）从 PyTorch 转换为 TensorFlow Lite（TFLite）格式，并集成到 Flutter 应用进行语音识别的详细指南。流程包括配置环境、验证依赖、运行单元测试、转换模型和集成到 Flutter，用于从 16kHz 单声道 WAV 音频提取 512 维语音嵌入并计算余弦相似度。本手册提供了 `StreamAudioDetector` 的完整技术文档和使用案例，帮助开发者快速上手和深入理解音频检测技术。

- **目标**：将 x-vector 模型转换为 TFLite，用于 Flutter 应用提取语音嵌入并进行语音匹配。
- **输入**：16kHz 单声道 WAV 音频（5-10 秒），处理为 MFCC 特征（`n_mels=24`，形状 `[1, time, 24]`）。
- **输出**：TFLite 模型（`x_vector.tflite`），生成 512 维嵌入（`[1, 512]`）。
- **环境**：Windows，Miniconda，Python 3.11，基于 CPU。
- **最终输出**：`x_vector.tflite` 和 Flutter `VoiceMatch` 类。

### 前提条件
- **硬件**：Windows PC，建议 8GB 以上内存。
- **软件**：
  - Miniconda（[conda.io](https://docs.conda.io/en/latest/miniconda.html)）。
  - Python 3.11。
  - Flutter SDK（[flutter.dev](https://flutter.dev/docs/get-started/install)）。
- **输入音频**：16kHz 单声道 WAV 文件（5-10 秒，例如 `sample.wav`）。
- **预训练模型**：SpeechBrain x-vector 模型（`spkrec-xvect-voxceleb`），从 [Hugging Face](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) 下载。
- **可选**：LibriSpeech 数据集用于额外测试（[openslr.org/12](http://www.openslr.org/12)）。

### 详细步骤

#### 步骤 1：配置 Conda 环境和安装依赖
创建 Conda 环境并安装依赖。

```bash
# 创建 Conda 环境
conda create -n speechbrain python=3.11

# 激活环境
conda activate speechbrain

# 安装依赖
D:\miniconda3\envs\speechbrain\python.exe -m pip install \
    speechbrain==1.0.3 \
    soundfile==0.13.1 \
    tensorflow==2.14.1 \
    onnx==1.16.0 \
    onnx_tf==1.10.0 \
    tensorflow-addons==0.22.0 \
    tensorflow_probability==0.22.1 \
    onnxruntime==1.18.0 \
    transformers \
    sounddevice \
    pytest \
    -i https://mirrors.aliyun.com/pypi/simple/ -vvv
```

**注意**：
- 使用 PyPI 镜像（如阿里云）加速下载。
- 确保使用指定版本避免兼容性问题。

#### 步骤 2：验证依赖
创建 `verify_install.py` 检查模块导入（文件位于仓库中）。

运行验证：
```bash
D:\miniconda3\envs\speechbrain\python.exe verify_install.py
```

**预期输出**：`All modules imported successfully!`

#### 步骤 3：验证 SpeechBrain 和模型转换
根据 [SpeechBrain 文档](https://speechbrain.readthedocs.io/en/latest/installation.html)，运行单元测试验证核心功能。

```bash
cd D:\speechbrain_project
D:\miniconda3\envs\speechbrain\python.exe -m pip install pytest -i https://mirrors.aliyun.com/pypi/simple/ -vvv
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain
D:\miniconda3\envs\speechbrain\python.exe -m pytest tests
```

**注意**：
- 测试可能需要额外依赖，可能因缺少数据集而部分失败，关注核心功能。

#### 步骤 4：运行 x-vector 模型转换
从 [Hugging Face](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) 下载模型到 `D:\speechbrain_project\pretrained_models\spkrec-xvect-voxceleb`。

准备 16kHz 单声道 WAV 文件（例如 `D:\speechbrain_project\sample.wav`）：
```bash
ffmpeg -i input.wav -ar 16000 -ac 1 -y D:\speechbrain_project\sample.wav
```

运行转换脚本（`convert_xvector.py`，位于仓库中）：
```bash
cd D:\speechbrain_project
D:\miniconda3\envs\speechbrain\python.exe convert_xvector.py
```

**输入**：`D:/speechbrain_project/sample.wav`

**预期输出**：
```
验证包版本...
speechbrain: 1.0.3
torch: 2.7.1+cpu
torchaudio: 2.7.1+cpu
onnx: 1.16.0
onnx_tf: 1.10.0
tensorflow: 2.14.1
numpy: 1.26.4
soundfile: 0.13.1
GPU Available: False
加载 x-vector 模型...
请提供一个 16kHz 单声道 WAV 文件（5-10 秒）。
输入 sample.wav 的路径（例如 D:\speechbrain_project\sample.wav）：D:/speechbrain_project/sample.wav
提取 MFCC...
MFCC 形状: torch.Size([1, 1089, 24])
导出模型到 ONNX...
验证 ONNX 模型...
ONNX 输出形状: (1, 512)
将 ONNX 转换为 TensorFlow...
转换为 TFLite...
验证 TFLite 模型...
TFLite 输入详情: [{'name': 'mfcc', 'index': 0, 'shape': ..., 'dtype': <class 'numpy.float32'>, ...}]
TFLite 输出详情: [{'name': 'embedding', 'index': ..., 'shape': [1, 512], 'dtype': <class 'numpy.float32'>, ...}]
TFLite 输出形状: (1, 512)
```

**输出文件**：
- `x_vector.onnx`
- `x_vector_pb`（目录）
- `x_vector.tflite`

#### 步骤 5：LibriSpeech 数据转换
使用 LibriSpeech 数据集进行额外测试。

1. **下载 LibriSpeech**：
   ```bash
   cd D:\speechbrain_project
   mkdir librispeech
   wget http://www.openslr.org/resources/12/dev-clean.tar.gz
   tar -xzvf dev-clean.tar.gz -C librispeech
   ```

2. **转换为 16kHz 单声道**：
   ```bash
   ffmpeg -i librispeech/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac -ar 16000 -ac 1 -y D:\speechbrain_project\sample_librispeech.wav
   ```

3. **使用 LibriSpeech 测试**：
   更新 `convert_xvector.py` 输入为 `D:/speechbrain_project/sample_librispeech.wav`，重新运行：
   ```bash
   D:\miniconda3\envs\speechbrain\python.exe convert_xvector.py
   ```

#### 步骤 6：集成到 Flutter
在 `pubspec.yaml` 中添加：
```yaml
dependencies:
  tflite_flutter: ^0.10.4
  flutter_sound: ^9.2.13
  flutter_sound_processing: ^0.0.2

assets:
  - assets/x_vector.tflite
  - assets/sample.wav
```

安装依赖：
```bash
cd <Flutter_project>
flutter pub get
```

使用仓库中的 `voice_match.dart` 进行嵌入提取和余弦相似度计算。

复制文件：
```bash
copy D:\speechbrain_project\x_vector.tflite <Flutter_project>\assets\x_vector.tflite
copy D:\speechbrain_project\sample.wav <Flutter_project>\assets\sample.wav
```

运行 Flutter：
```bash
flutter run
```

### 故障排除
- **无效输入 'lengths'**：在 `XVectorWrapper` 中固定 `lengths=torch.ones(batch_size)`。
- **MFCC 通道不匹配**：设置 `Fbank(n_mels=24)`。
- **ONNX 错误**（`LeakyRelu`、`Unsqueeze`）：使用 `opset_version=11`。
- **TFLite 错误**（`tf.RandomStandardNormal`）：启用 `SELECT_TF_OPS`，禁用优化，设置 `model.eval()`。
- **动态形状**：使用 `resize_tensor_input` 处理可变 MFCC 长度。

### 验证
比较输出：
```python
pytorch_output = model_wrapper(mfcc).detach().numpy()
print(f"PyTorch 输出: {pytorch_output}")
print(f"ONNX 输出: {onnx_output}")
print(f"TFLite 输出: {tflite_output}")
print(f"PyTorch vs ONNX 差值: {np.abs(pytorch_output - onnx_output).max()}")
print(f"ONNX vs TFLite 差值: {np.abs(onnx_output - tflite_output).max()}")
```

检查 ONNX 输入：
```python
import onnx
onnx_model = onnx.load("x_vector.onnx")
for input in onnx_model.graph.input:
    print(input.name)  # 应为 'mfcc'
```

### 注意事项
- 使用指定依赖版本。
- `model.eval()` 禁用 dropout。
- 动态形状支持可变 MFCC 输入。
- 在目标设备上测试性能。

### 仓库结构
```
speechbrain-xvector-tflite/
├── convert_xvector.py
├── verify_install.py
├── pretrained_models/spkrec-xvect-voxceleb/
├── sample.wav
├── librispeech/ (可选)
├── flutter_project/
│   ├── lib/voice_match.dart
│   ├── assets/x_vector.tflite
│   ├── assets/sample.wav
├── README.md
├── LICENSE
└── requirements.txt
```

**requirements.txt**：
```
speechbrain==1.0.3
soundfile==0.13.1
tensorflow==2.14.1
onnx==1.16.0
onnx_tf==1.10.0
tensorflow-addons==0.22.0
tensorflow_probability==0.22.1
onnxruntime==1.18.0
transformers
sounddevice
pytest
```

**版权声明**  
如果您参考或使用本代码，请注明出处：  
GitHub: @didadidaboom  
© 2024 @didadidaboom. 保留所有权利。  
本作品采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

---

## English Version

### Project Overview
This project provides a detailed guide to convert the SpeechBrain x-vector model (`spkrec-xvect-voxceleb`) from PyTorch to TensorFlow Lite (TFLite) format and integrate it into a Flutter application for speaker recognition. The process includes setting up the environment, verifying dependencies, running unit tests, converting the model, and integrating it into Flutter to extract 512-dimensional speaker embeddings and compute cosine similarity for voice matching. This manual provides comprehensive technical documentation and use cases for `StreamAudioDetector`, helping developers quickly get started and deeply understand audio detection technology.

- **Objective**: Convert the x-vector model to TFLite for use in a Flutter app to extract embeddings and perform voice matching.
- **Input**: 16kHz mono WAV audio (5-10 seconds), processed to MFCC features (`n_mels=24`, shape `[1, time, 24]`).
- **Output**: TFLite model (`x_vector.tflite`) producing 512-dimensional embeddings (`[1, 512]`).
- **Environment**: Windows, Miniconda, Python 3.11, CPU-based setup.
- **Final Output**: `x_vector.tflite` and a Flutter `VoiceMatch` class.

### Prerequisites
- **Hardware**: Windows PC with 8GB+ RAM.
- **Software**:
  - Miniconda ([conda.io](https://docs.conda.io/en/latest/miniconda.html)).
  - Python 3.11.
  - Flutter SDK ([flutter.dev](https://flutter.dev/docs/get-started/install)).
- **Input Audio**: 16kHz mono WAV file (5-10 seconds, e.g., `sample.wav`).
- **Pretrained Model**: SpeechBrain x-vector model (`spkrec-xvect-voxceleb`) from [Hugging Face](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb).
- **Optional**: LibriSpeech dataset for additional testing ([openslr.org/12](http://www.openslr.org/12)).

### Step-by-Step Guide

#### Step 1: Set Up Conda Environment and Install Dependencies
Create a Conda environment and install required packages.

```bash
# Create Conda environment
conda create -n speechbrain python=3.11

# Activate environment
conda activate speechbrain

# Install dependencies
D:\miniconda3\envs\speechbrain\python.exe -m pip install \
    speechbrain==1.0.3 \
    soundfile==0.13.1 \
    tensorflow==2.14.1 \
    onnx==1.16.0 \
    onnx_tf==1.10.0 \
    tensorflow-addons==0.22.0 \
    tensorflow_probability==0.22.1 \
    onnxruntime==1.18.0 \
    transformers \
    sounddevice \
    pytest \
    -i https://mirrors.aliyun.com/pypi/simple/ -vvv
```

**Notes**:
- Use a PyPI mirror (e.g., Aliyun) for faster downloads.
- Ensure exact versions to avoid compatibility issues.

#### Step 2: Verify Dependencies
Use `verify_install.py` (in the repository) to check module imports.

Run verification:
```bash
D:\miniconda3\envs\speechbrain\python.exe verify_install.py
```

**Expected Output**: `All modules imported successfully!`

#### Step 3: Verify SpeechBrain and Model Conversion
Run SpeechBrain unit tests to ensure core functionality, as per the [SpeechBrain documentation](https://speechbrain.readthedocs.io/en/latest/installation.html).

```bash
cd D:\speechbrain_project
D:\miniconda3\envs\speechbrain\python.exe -m pip install pytest -i https://mirrors.aliyun.com/pypi/simple/ -vvv
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain
D:\miniconda3\envs\speechbrain\python.exe -m pytest tests
```

**Notes**:
- Tests may require additional dependencies and may partially fail due to missing datasets; focus on core functionality.

#### Step 4: Run x-vector Model Conversion
Download the pretrained model to `D:\speechbrain_project\pretrained_models\spkrec-xvect-voxceleb` from [Hugging Face](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb).

Prepare a 16kHz mono WAV file (e.g., `D:\speechbrain_project\sample.wav`):
```bash
ffmpeg -i input.wav -ar 16000 -ac 1 -y D:\speechbrain_project\sample.wav
```

Run the conversion script (`convert_xvector.py`, in the repository):
```bash
cd D:\speechbrain_project
D:\miniconda3\envs\speechbrain\python.exe convert_xvector.py
```

**Input**: `D:/speechbrain_project/sample.wav`

**Expected Output**:
```
验证包版本...
speechbrain: 1.0.3
torch: 2.7.1+cpu
torchaudio: 2.7.1+cpu
onnx: 1.16.0
onnx_tf: 1.10.0
tensorflow: 2.14.1
numpy: 1.26.4
soundfile: 0.13.1
GPU Available: False
加载 x-vector 模型...
请提供一个 16kHz 单声道 WAV 文件（5-10 秒）。
输入 sample.wav 的路径（例如 D:\speechbrain_project\sample.wav）：D:/speechbrain_project/sample.wav
提取 MFCC...
MFCC 形状: torch.Size([1, 1089, 24])
导出模型到 ONNX...
验证 ONNX 模型...
ONNX 输出形状: (1, 512)
将 ONNX 转换为 TensorFlow...
转换为 TFLite...
验证 TFLite 模型...
TFLite 输入详情: [{'name': 'mfcc', 'index': 0, 'shape': ..., 'dtype': <class 'numpy.float32'>, ...}]
TFLite 输出详情: [{'name': 'embedding', 'index': ..., 'shape': [1, 512], 'dtype': <class 'numpy.float32'>, ...}]
TFLite 输出形状: (1, 512)
```

**Output Files**:
- `x_vector.onnx`
- `x_vector_pb` (directory)
- `x_vector.tflite`

#### Step 5: LibriSpeech Data Conversion
Use the LibriSpeech dataset for additional testing ([openslr.org/12](http://www.openslr.org/12)).

1. **Download LibriSpeech**:
   ```bash
   cd D:\speechbrain_project
   mkdir librispeech
   wget http://www.openslr.org/resources/12/dev-clean.tar.gz
   tar -xzvf dev-clean.tar.gz -C librispeech
   ```

2. **Convert to 16kHz Mono**:
   ```bash
   ffmpeg -i librispeech/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac -ar 16000 -ac 1 -y D:\speechbrain_project\sample_librispeech.wav
   ```

3. **Test with LibriSpeech**:
   Update `convert_xvector.py` input to `D:/speechbrain_project/sample_librispeech.wav` and rerun:
   ```bash
   D:\miniconda3\envs\speechbrain\python.exe convert_xvector.py
   ```

#### Step 6: Integrate with Flutter
Add to `pubspec.yaml`:
```yaml
dependencies:
  tflite_flutter: ^0.10.4
  flutter_sound: ^9.2.13
  flutter_sound_processing: ^0.0.2

assets:
  - assets/x_vector.tflite
  - assets/sample.wav
```

Install dependencies:
```bash
cd <Flutter_project>
flutter pub get
```

Use `voice_match.dart` (in the repository) for embedding extraction and cosine similarity.

Copy files:
```bash
copy D:\speechbrain_project\x_vector.tflite <Flutter_project>\assets\x_vector.tflite
copy D:\speechbrain_project\sample.wav <Flutter_project>\assets\sample.wav
```

Run Flutter:
```bash
flutter run
```

### Troubleshooting
- **Invalid Input 'lengths'**: Fixed in `XVectorWrapper` with `lengths=torch.ones(batch_size)`.
- **MFCC Channel Mismatch**: Set `Fbank(n_mels=24)`.
- **ONNX Errors** (`LeakyRelu`, `Unsqueeze`): Used `opset_version=11`.
- **TFLite Error** (`tf.RandomStandardNormal`): Enabled `SELECT_TF_OPS`, disabled optimizations, set `model.eval()`.
- **Dynamic Shapes**: Used `resize_tensor_input` for variable MFCC lengths.

### Validation
Compare outputs:
```python
pytorch_output = model_wrapper(mfcc).detach().numpy()
print(f"PyTorch Output: {pytorch_output}")
print(f"ONNX Output: {onnx_output}")
print(f"TFLite Output: {tflite_output}")
print(f"PyTorch vs ONNX Diff: {np.abs(pytorch_output - onnx_output).max()}")
print(f"ONNX vs TFLite Diff: {np.abs(onnx_output - tflite_output).max()}")
```

Check ONNX inputs:
```python
import onnx
onnx_model = onnx.load("x_vector.onnx")
for input in onnx_model.graph.input:
    print(input.name)  # Should be 'mfcc'
```

### Notes
- Use specified dependency versions.
- `model.eval()` prevents random operations (e.g., dropout).
- Dynamic shapes support variable-length MFCC inputs.
- Test on target devices for performance.

### Repository Structure
```
speechbrain-xvector-tflite/
├── convert_xvector.py
├── verify_install.py
├── pretrained_models/spkrec-xvect-voxceleb/
├── sample.wav
├── librispeech/ (optional)
├── flutter_project/
│   ├── lib/voice_match.dart
│   ├── assets/x_vector.tflite
│   ├── assets/sample.wav
├── README.md
├── LICENSE
└── requirements.txt
```

**requirements.txt**:
```
speechbrain==1.0.3
soundfile==0.13.1
tensorflow==2.14.1
onnx==1.16.0
onnx_tf==1.10.0
tensorflow-addons==0.22.0
tensorflow_probability==0.22.1
onnxruntime==1.18.0
transformers
sounddevice
pytest
```

**Copyright Notice**  
If you reference or use this code, please cite the source:  
GitHub: @didadidaboom  
© 2024 @didadidaboom. All rights reserved.  
This work is licensed under the MIT License. See [LICENSE](LICENSE) file for details.
