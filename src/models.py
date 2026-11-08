
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

if __name__ == "__main__":

    # 测试 MLP
    mlp = MLP()

    # 测试 CNN
    cnn = CNN()

    # 构造一个模拟批次：64张28×28灰度图片
    dummy_input = ms.ops.zeros(
        (64, 1, 28, 28),
        ms.float32
    )

    # 前向传播
    output = cnn(dummy_input)

    print("CNN 输出形状：", output.shape)

    # 统计参数量
    parameter_count = sum(
        p.size for p in cnn.trainable_params()
    )

    print("CNN 参数数量：", parameter_count)

    print("\n各层参数量：")

    for name, parameter in cnn.parameters_and_names():
        print(name, parameter.shape, parameter.size)