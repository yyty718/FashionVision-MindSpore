
# FashionVision-MindSpore

**基于 MindSpore 的 Fashion-MNIST 图像分类项目**

本项目使用 Python 和 MindSpore 深度学习框架，
从零实现一个 Fashion-MNIST 图像分类系统。

项目将从数据集读取与预处理开始，逐步实现不同类型的
神经网络模型，并通过实验比较不同模型的分类效果。

目前项目正在开发中。

## 一、项目介绍

Fashion-MNIST 是一个常用的图像分类数据集，
包含 10 种不同类别的服饰灰度图像。

本项目计划通过实现多种神经网络模型，
学习深度学习图像分类任务的基本流程，
理解数据预处理、模型构建、模型训练和模型评估等环节。

## 二、数据集介绍

使用 Fashion-MNIST 数据集。

- 训练集：60,000 张图像
- 测试集：10,000 张图像
- 图像尺寸：28 × 28
- 图像通道：单通道灰度图
- 分类类别：10 类

## 三、技术栈

- Python 3.11
- MindSpore 2.10.0
- NumPy
- Matplotlib
- Scikit-learn
- PyCharm
- Git / GitHub

## 四、模型设计

计划实现并比较以下三种模型：

### 1. 多层感知机（MLP）

使用全连接神经网络完成图像分类，
作为基础模型和实验对照。

### 2. 基础卷积神经网络（CNN）

使用卷积层和池化层提取图像特征，
研究卷积神经网络在图像分类任务中的表现。

### 3. 改进卷积神经网络（Improved CNN）

在基础 CNN 的基础上引入 Batch Normalization、
Dropout 等方法，尝试改善模型的分类效果。

## 五、项目计划

- [ ] 创建项目目录与 Python 虚拟环境
- [ ] 配置 Git 并完成首次 GitHub 推送
- [ ] 编写项目说明与依赖清单
- [ ] Fashion-MNIST 数据集读取与预处理
- [ ] MLP 模型实现
- [ ] 基础 CNN 模型实现
- [ ] Improved CNN 模型实现
- [ ] 模型训练与参数调整
- [ ] 模型评价与实验对比
- [ ] 结果可视化与单张图片预测
- [ ] 完善实验分析与项目文档

## 六、项目结构

随着项目开发逐步完善，预计采用以下结构：

```text
FashionVision-MindSpore/
├── data/                   # 数据集
├── models/                 # 模型文件
├── results/                # 实验结果
├── src/                    # 核心源代码
├── .gitignore
├── requirements.txt
├── README.md
└── main.py
```

## 七、项目状态

当前阶段：项目初始化已完成，准备进行数据集读取与预处理。

本项目将持续记录开发过程、实验结果和阶段性总结。
