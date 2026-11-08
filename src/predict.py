
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import mindspore as ms

from dataset import create_dataset
from models import ImprovedCNNWithoutDropout


# =========================
# 1. 项目路径
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints" / "improved_cnn_no_dropout.ckpt"
)


# =========================
# 2. Fashion-MNIST 类别名称
# =========================

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


# =========================
# 3. 加载测试集
# =========================

def get_one_test_sample():

    test_dataset = create_dataset(
        usage="test",
        shuffle=False,
        batch_size=64
    )

    # 读取测试集中的第一个批次
    data_iterator = test_dataset.create_dict_iterator(
        output_numpy=True
    )

    batch = next(data_iterator)

    # 取出第一个样本
    image = batch["image"][0:1]
    label = int(batch["label"][0])

    return image, label



# =========================
# 4. 加载模型并进行预测
# =========================

def predict_one(image):

    # 创建模型
    network = ImprovedCNNWithoutDropout()

    # 加载训练好的模型权重
    param_dict = ms.load_checkpoint(str(CHECKPOINT_PATH))
    ms.load_param_into_net(network, param_dict)

    # 切换到推理模式
    network.set_train(False)

    # 将 NumPy 图片转换成 MindSpore Tensor
    image_tensor = ms.Tensor(image, ms.float32)

    # 模型前向传播，得到原始分类分数
    logits = network(image_tensor)

    # 使用 Softmax 转换为概率
    probabilities = ms.ops.softmax(logits, axis=1)

    # 转换为 NumPy 数组
    probabilities = probabilities.asnumpy()[0]

    # 找到概率最大的类别
    predicted_label = int(np.argmax(probabilities))

    return predicted_label, probabilities



# =========================
# 5. 可视化预测结果
# =========================

def visualize_prediction(image, true_label, predicted_label, probabilities):

    # 还原图片像素值
    # 前面预处理公式： (原始像素 / 255 - 0.2860) / 0.3530
    # 这里进行反向计算，恢复到 [0, 1] 范围
    display_image = image[0, 0] * 0.3530 + 0.2860
    display_image = np.clip(display_image, 0, 1)

    # 创建画布：左边显示图片，右边显示概率
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左侧：原始服装图片
    axes[0].imshow(display_image, cmap="gray")
    axes[0].axis("off")
    axes[0].set_title(
        f"True: {CLASS_NAMES[true_label]}\n"
        f"Predicted: {CLASS_NAMES[predicted_label]}",
        fontsize=13
    )

    # 右侧：各类别预测概率
    axes[1].barh(CLASS_NAMES, probabilities * 100)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Probability (%)")
    axes[1].set_title("Prediction Probabilities")
    axes[1].set_xlim(0, 100)

    # 在柱状图上标注概率
    for i, probability in enumerate(probabilities):
        axes[1].text(
            probability * 100 + 0.5,
            i,
            f"{probability * 100:.2f}%",
            va="center",
            fontsize=9
        )

    # 总标题
    result = "Correct" if predicted_label == true_label else "Incorrect"

    fig.suptitle(
        f"Fashion-MNIST Prediction: {result}",
        fontsize=16
    )

    plt.tight_layout()

    # 保存图片
    save_path = PROJECT_ROOT / "results" / "prediction_result.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print("\n预测可视化图片已保存：", save_path)


# =========================
# 6. 测试图片读取
# =========================


if __name__ == "__main__":

    # 读取一张测试图片
    image, true_label = get_one_test_sample()

    print("图片读取成功！")
    print("图片形状：", image.shape)
    print("图片数据类型：", image.dtype)

    # 使用模型进行预测
    predicted_label, probabilities = predict_one(image)

    print("\n========== 预测结果 ==========")
    print("真实类别：", CLASS_NAMES[true_label])
    print("预测类别：", CLASS_NAMES[predicted_label])
    print("预测是否正确：", predicted_label == true_label)

    print("\n各类别预测概率：")

    for i, probability in enumerate(probabilities):
        print(
            f"{CLASS_NAMES[i]:15s}: {probability * 100:.2f}%"
        )

    print("==============================")

    # 显示并保存预测可视化结果
    visualize_prediction(
        image,
        true_label,
        predicted_label,
        probabilities
    )

