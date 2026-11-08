
"""
FashionVision-MindSpore
MLP 模型评估模块
"""

from pathlib import Path

import mindspore as ms

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from dataset import create_dataset
from models import ImprovedCNNWithoutDropout


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 模型参数文件
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "improved_cnn_no_dropout.ckpt"

# Fashion-MNIST 类别名称
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


def evaluate():
    """评估训练好的 MLP 模型"""

    # 1. 设置运行环境
    ms.set_context(
        mode=ms.PYNATIVE_MODE,
        device_target="CPU"
    )

    # 2. 加载测试集
    test_dataset = create_dataset(
        usage="test",
        shuffle=False,
        batch_size=64
    )

    print("测试集加载完成！")
    print("测试集批次数量：", test_dataset.get_dataset_size())

    # 3. 创建 CNN 网络
    network = ImprovedCNNWithoutDropout()

    # 4. 加载训练好的模型参数
    param_dict = ms.load_checkpoint(str(CHECKPOINT_PATH))

    ms.load_param_into_net(network, param_dict)

    # 5. 切换到评估模式
    network.set_train(False)

    print("\n模型参数加载完成！")
    print("开始评估 Improved CNN Without Dropout模型！\n")

    # 6. 保存真实标签和预测标签
    all_labels = []
    all_predictions = []

    # 7. 遍历测试集
    for batch in test_dataset.create_dict_iterator():

        images = batch["image"]
        labels = batch["label"]

        # 模型输出：每张图片对应10个类别的得分
        outputs = network(images)

        # 取分数最高的类别作为预测结果
        predictions = outputs.asnumpy().argmax(axis=1)

        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.asnumpy().tolist())

    # 8. 计算准确率
    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    print("=" * 50)
    print("Improved CNN Without Dropout模型测试结果")
    print("=" * 50)

    print(f"\nAccuracy（准确率）：{accuracy:.4f}")
    print(f"Accuracy（百分比）：{accuracy * 100:.2f}%")

    # 9. 输出每个类别的评价指标
    print("\n各类别 Precision、Recall 和 F1-score：")

    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=list(range(10)),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0
        )
    )

    # 10. 计算混淆矩阵
    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(10))
    )

    print("混淆矩阵（Confusion Matrix）：")
    print(matrix)


if __name__ == "__main__":
    evaluate()
