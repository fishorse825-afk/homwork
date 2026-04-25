# 眼底图像血管分割项目

基于深度学习的2D眼底图像血管分割任务，用于Kaggle比赛提交。

## 项目概述

本项目使用UNet深度学习模型完成眼底图像中的血管分割任务，包含完整的训练、预测和提交流程。

### 技术栈

- Python 3.10+
- PyTorch 2.x
- Albumentations（数据增强）
- PIL/Pandas/Numpy

## 项目结构

```
Vascular segmentation of fundus images/
├── train/                    # 训练集（600对图像-标注）
│   ├── image/              # 眼底图像
│   └── label/              # 血管标注mask
├── test/                     # 测试集（112张图像）
│   └── image/
├── predictions/              # 预测结果（生成）
├── model.py                 # UNet模型定义
├── dataset.py               # 数据集加载器
├── train.py                 # 训练脚本
├── predict.py               # 预测脚本
├── segmentation_to_csv.py   # 生成提交CSV
├── best_model.pth           # 训练好的模型（生成）
└── sample_submission.csv    # Kaggle提交文件（生成）
```

## 安装依赖

```bash
# 创建虚拟环境
conda create -n ai-infra python=3.10
conda activate ai-infra

# 安装核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pillow pandas albumentations
```

## 使用说明

### 1. 训练模型

```bash
cd "Vascular segmentation of fundus images"
python train.py
```

训练配置：

- 批大小：4
- 学习率：1e-4
- Epoch数：50
- 损失函数：Dice Loss

### 2. 生成预测

```bash
python predict.py
```

预测结果保存在 `predictions/` 目录。

### 3. 生成提交文件

```bash
python segmentation_to_csv.py
```

生成 `sample_submission.csv` 文件，用于Kaggle提交。

## 模型架构

采用经典UNet架构，包含：

- **编码器**：4层下采样，通道数从64增加到512
- **解码器**：4层上采样，通过跳跃连接融合特征
- **跳跃连接**：融合高层语义信息和低层细节信息

## 数据增强

- 随机旋转（±15°）
- 随机水平翻转
- 随机垂直翻转
- ImageNet归一化
