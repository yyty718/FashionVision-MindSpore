
"""
FashionVision-MindSpore
神经网络模型定义
"""

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

if __name__ == "__main__":

    import mindspore as ms

    # 创建 MLP 模型
    model = MLP()

    # 模拟一个批次的图片
    dummy_input = ms.Tensor(
        shape=(64, 1, 28, 28),
        dtype=ms.float32
    )

    # 执行前向传播
    output = model(dummy_input)

    print("输入形状：", dummy_input.shape)
    print("输出形状：", output.shape)
    print("模型参数数量：", sum(
        parameter.size for parameter in model.get_parameters()
    ))