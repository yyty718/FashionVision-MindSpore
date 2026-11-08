
"""
FashionVision-MindSpore
MLP 模型训练模块
"""

from pathlib import Path

import mindspore as ms
import mindspore.nn as nn
from mindspore.train import Model
from mindspore.train.callback import LossMonitor

from dataset import create_dataset
from models import MLP,CNN,ImprovedCNN,ImprovedCNNWithoutDropout


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 模型保存目录
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def train():
    """训练 MLP 模型"""

    # 1. 设置运行环境
    ms.set_context(
        mode=ms.PYNATIVE_MODE,
        device_target="CPU"
    )

    # 2. 加载训练集
    train_dataset = create_dataset(
        usage="train",
        shuffle=True,
        batch_size=64
    )

    print("训练集加载完成！")
    print("训练集批次数量：", train_dataset.get_dataset_size())

    # 3. 创建 CNN 网络
    network = ImprovedCNNWithoutDropout()

    # 4. 定义损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 5. 定义优化器
    optimizer = nn.Adam(
        params=network.trainable_params(),
        learning_rate=0.001
    )

    # 6. 组合网络、损失函数和优化器
    model = Model(
        network=network,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics={"Accuracy": nn.Accuracy()}
    )

    # 7. 设置损失监控
    loss_monitor = LossMonitor(per_print_times=100)

    # 8. 开始训练
    print("\n开始训练 Improved CNN Without Dropout 模型！")

    model.train(
        5,
        train_dataset,
        callbacks=[loss_monitor],
        dataset_sink_mode=False
    )

    print("\nImproved CNN Without Dropout 模型训练完成！")

    # 9. 保存训练好的模型
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = CHECKPOINT_DIR / "improved_cnn_no_dropout.ckpt"

    ms.save_checkpoint(
        network,
        str(checkpoint_path)
    )

    print("\n模型参数已保存！")
    print("保存位置：", checkpoint_path)


if __name__ == "__main__":
    train()
