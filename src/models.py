
"""
FashionVision-MindSpore
神经网络模型定义
"""

import mindspore as ms
import mindspore.nn as nn
# nn.Cell MindSpore中定义神经网络的基础类。
# 我们创建的 MLP 继承它，才能成为 MindSpore 可以执行的网络模型


class MLP(nn.Cell):
    """
    多层感知机（Multi-Layer Perceptron）
    用于 Fashion-MNIST 图像分类
    """

    def __init__(self): #负责定义网络有哪些层
        super(MLP, self).__init__()

        # 将 28×28 的图片展平为 784 维向量
        self.flatten = nn.Flatten()

        # 第一层全连接层：784 → 256
        self.fc1 = nn.Dense(784, 256)

        # ReLU 激活函数
        self.relu1 = nn.ReLU()

        # 第二层全连接层：256 → 128
        self.fc2 = nn.Dense(256, 128)

        # ReLU 激活函数
        self.relu2 = nn.ReLU()

        # 输出层：128 → 10
        self.fc3 = nn.Dense(128, 10)

    def construct(self, x):
        """定义数据在网络中的前向传播过程
            即定义数据如何经过这些层"""

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)

        return x

class CNN(nn.Cell):
    """基础卷积神经网络"""

    def __init__(self):
        super(CNN, self).__init__()

        # 第一组卷积层
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            pad_mode="same",
            has_bias=True
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # 第二组卷积层
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            pad_mode="same",
            has_bias=True
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # 全连接分类层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Dense(128, 10)

    def construct(self, x):
        """定义前向传播过程
           即定义数据经过这些层的顺序
           顺序很重要，不能随意交换！"""

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x

class ImprovedCNN(nn.Cell):
    """改进型卷积神经网络"""

    def __init__(self):
        super(ImprovedCNN, self).__init__()

        # 第一组卷积模块
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            pad_mode="same",
            has_bias=True
        )
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2d_1 = nn.Dropout2d(p=0.25)

        # 第二组卷积模块
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            pad_mode="same",
            has_bias=True
        )
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2d_2 = nn.Dropout2d(p=0.25)

        # 第三组卷积模块
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            pad_mode="same",
            has_bias=True
        )
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.relu3 = nn.ReLU()

        # 全连接分类模块
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(64 * 7 * 7, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Dense(128, 10)

    def construct(self, x):
        """定义前向传播过程"""

        # 第一组卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout2d_1(x)

        # 第二组卷积
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2d_2(x)

        # 第三组卷积
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # 全连接分类
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class ImprovedCNNWithoutDropout(nn.Cell):
    """去掉 Dropout 的改进型卷积神经网络"""

    def __init__(self):
        super(ImprovedCNNWithoutDropout, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, pad_mode="same", has_bias=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, pad_mode="same", has_bias=True)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, pad_mode="same", has_bias=True)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(64 * 7 * 7, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Dense(128, 10)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = ImprovedCNNWithoutDropout()
    dummy_input = ms.ops.zeros((64, 1, 28, 28), ms.float32)

    output = model(dummy_input)

    print("Improved CNN Without Dropout 输出形状：", output.shape)

    parameter_count = sum(p.size for p in model.trainable_params())
    print("参数数量：", parameter_count)

"""
if __name__ == "__main__":

    # 创建三个模型
    mlp = MLP()
    cnn = CNN()
    improved_cnn = ImprovedCNN()

    # 构造一个模拟批次
    dummy_input = ms.ops.zeros(
        (64, 1, 28, 28),
        ms.float32
    )

    # 测试 Improved CNN
    output = improved_cnn(dummy_input)

    print("Improved CNN 输出形状：", output.shape)

    # 统计参数数量
    parameter_count = sum(
        p.size for p in improved_cnn.trainable_params()
    )

    print("Improved CNN 参数数量：", parameter_count)
"""