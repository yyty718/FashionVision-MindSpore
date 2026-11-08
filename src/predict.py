
import numpy as np
import matplotlib

# 使用无窗口绘图模式，避免 Windows 下绘图窗口卡死
matplotlib.use("Agg")

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

RESULTS_DIR = PROJECT_ROOT / "results"


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
# 3. 加载完整测试集
# =========================

def get_test_data():

    test_dataset = create_dataset(
        usage="test",
        shuffle=False,
        batch_size=64
    )

    images = []
    labels = []

    for batch in test_dataset.create_dict_iterator(output_numpy=True):
        images.append(batch["image"])
        labels.append(batch["label"])

    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)

    return images, labels


# =========================
# 4. 加载训练好的模型
# =========================

def load_model():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"找不到模型权重文件：{CHECKPOINT_PATH}"
        )

    network = ImprovedCNNWithoutDropout()

    param_dict = ms.load_checkpoint(str(CHECKPOINT_PATH))
    ms.load_param_into_net(network, param_dict)

    network.set_train(False)

    return network


# =========================
# 5. 对完整测试集进行预测
# =========================

def predict_all(network, images):

    all_probabilities = []
    batch_size = 256

    for start in range(0, len(images), batch_size):

        end = min(start + batch_size, len(images))

        batch_images = images[start:end]

        image_tensor = ms.Tensor(
            batch_images,
            ms.float32
        )

        logits = network(image_tensor)

        probabilities = ms.ops.softmax(
            logits,
            axis=1
        ).asnumpy()

        all_probabilities.append(probabilities)

        print(f"预测进度：{end}/{len(images)}")

    all_probabilities = np.concatenate(
        all_probabilities,
        axis=0
    )

    predicted_labels = np.argmax(
        all_probabilities,
        axis=1
    )

    return predicted_labels, all_probabilities


# =========================
# 6. 图片反归一化
# =========================

def restore_image(image):

    display_image = image[0] * 0.3530 + 0.2860

    return np.clip(display_image, 0, 1)


# =========================
# 7. 生成20张随机图片预测网格
# =========================

def visualize_prediction_grid(
    images,
    true_labels,
    predicted_labels
):

    # 关键修复：这里只展示20张，而不是整个测试集
    num_samples = min(20, len(images))

    rng = np.random.default_rng(42)

    indices = rng.choice(
        len(images),
        size=num_samples,
        replace=False
    )

    cols = 5
    rows = int(np.ceil(num_samples / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(15, 10)
    )

    axes = np.array(axes).flatten()

    correct_count = 0

    for i, index in enumerate(indices):

        true_label = int(true_labels[index])
        predicted_label = int(predicted_labels[index])

        is_correct = true_label == predicted_label

        if is_correct:
            correct_count += 1

        display_image = restore_image(images[index])

        axes[i].imshow(
            display_image,
            cmap="gray",
            interpolation="nearest",
            vmin=0,
            vmax=1
        )

        axes[i].set_xticks([])
        axes[i].set_yticks([])

        result = "Correct" if is_correct else "Incorrect"

        axes[i].set_title(
            f"True: {CLASS_NAMES[true_label]}\n"
            f"Pred: {CLASS_NAMES[predicted_label]}\n"
            f"{result}",
            fontsize=9,
            color="green" if is_correct else "red",
            pad=5
        )

    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    accuracy = correct_count / num_samples * 100

    fig.suptitle(
        "Fashion-MNIST Prediction Results",
        fontsize=17,
        fontweight="bold"
    )

    fig.text(
        0.5,
        0.94,
        f"Correct: {correct_count}/{num_samples}    "
        f"Accuracy: {accuracy:.2f}%",
        ha="center",
        fontsize=12
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    save_path = RESULTS_DIR / "prediction_grid.png"

    fig.savefig(
        save_path,
        dpi=200,
        facecolor="white"
    )

    plt.close(fig)

    print("20张图片预测结果已保存：", save_path)

    return indices


# =========================
# 8. 生成预测概率柱状图
# =========================

def visualize_prediction_probability(
    image,
    true_label,
    predicted_label,
    probabilities
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [1, 2]}
    )

    display_image = restore_image(image)

    axes[0].imshow(
        display_image,
        cmap="gray",
        interpolation="nearest",
        vmin=0,
        vmax=1
    )

    axes[0].axis("off")
    axes[0].set_title("Test Image")

    percentages = probabilities * 100

    bars = axes[1].barh(
        CLASS_NAMES,
        percentages
    )

    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 110)

    axes[1].set_xlabel("Prediction Probability (%)")
    axes[1].set_title("Prediction Probabilities")

    axes[1].grid(
        axis="x",
        linestyle="--",
        alpha=0.3
    )

    for i, bar in enumerate(bars):

        axes[1].text(
            percentages[i] + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{percentages[i]:.2f}%",
            va="center",
            fontsize=8
        )

    result = (
        "Correct"
        if true_label == predicted_label
        else "Incorrect"
    )

    fig.suptitle(
        f"True: {CLASS_NAMES[true_label]}    "
        f"Predicted: {CLASS_NAMES[predicted_label]}    "
        f"{result}",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    save_path = RESULTS_DIR / "prediction_probability.png"

    fig.savefig(
        save_path,
        dpi=200,
        facecolor="white"
    )

    plt.close(fig)

    print("预测概率图已保存：", save_path)


#
# =========================
# 9. 生成预测错误案例图
# =========================

def visualize_prediction_error(
    image,
    true_label,
    predicted_label,
    probabilities
):

    # 创建画布
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        gridspec_kw={"width_ratios": [1, 2]}
    )

    # =========================
    # 左侧：错误预测图片
    # =========================

    display_image = restore_image(image)

    axes[0].imshow(
        display_image,
        cmap="gray",
        interpolation="nearest",
        vmin=0,
        vmax=1
    )

    axes[0].axis("off")

    axes[0].set_title(
        "Test Image",
        fontsize=14,
        pad=12
    )

    # =========================
    # 右侧：10个类别的预测概率
    # =========================

    percentages = probabilities * 100

    bars = axes[1].barh(
        CLASS_NAMES,
        percentages
    )

    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 110)

    axes[1].set_xlabel(
        "Prediction Probability (%)",
        fontsize=11
    )

    axes[1].set_title(
        "Prediction Probabilities",
        fontsize=14,
        pad=12
    )

    axes[1].grid(
        axis="x",
        linestyle="--",
        alpha=0.3
    )

    axes[1].set_axisbelow(True)

    # 标注每个类别的概率
    for i, bar in enumerate(bars):

        axes[1].text(
            percentages[i] + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{percentages[i]:.2f}%",
            va="center",
            fontsize=9
        )

    # =========================
    # 顶部：真实类别和预测结果
    # =========================

    confidence = probabilities[predicted_label] * 100

    fig.suptitle(
        f"True: {CLASS_NAMES[true_label]}    "
        f"Predicted: {CLASS_NAMES[predicted_label]}    "
        f"Incorrect",
        fontsize=17,
        fontweight="bold",
        color="red",
        y=0.97
    )

    fig.text(
        0.5,
        0.91,
        f"Prediction Confidence: {confidence:.2f}%",
        ha="center",
        fontsize=12,
        color="dimgray"
    )

    # 预留顶部空间，避免标题被截断
    plt.subplots_adjust(
        left=0.07,
        right=0.96,
        top=0.83,
        bottom=0.10,
        wspace=0.30
    )

    # 保存图片
    save_path = RESULTS_DIR / "prediction_error.png"

    fig.savefig(
        save_path,
        dpi=200,
        facecolor="white",
        bbox_inches="tight"
    )

    plt.close(fig)

    print("错误案例图片及概率柱状图已保存：", save_path)


# =========================
# 10. 主程序
# =========================

if __name__ == "__main__":

    NUM_SAMPLES = 20

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("========== Fashion-MNIST 批量预测 ==========")

    # 加载测试集
    print("\n正在加载测试集……")

    images, true_labels = get_test_data()

    print("测试集图片数量：", len(images))
    print("图片形状：", images.shape)

    # 加载模型
    print("\n正在加载模型……")

    network = load_model()

    print("模型：ImprovedCNNWithoutDropout")
    print("权重文件：", CHECKPOINT_PATH)

    # 完整测试集预测
    print("\n正在对完整测试集进行预测……")

    predicted_labels, all_probabilities = predict_all(
        network,
        images
    )

    # 完整测试集统计
    correct_count = np.sum(
        predicted_labels == true_labels
    )

    total_accuracy = (
        correct_count / len(true_labels) * 100
    )

    print("\n========== 完整测试集结果 ==========")
    print("测试图片数量：", len(true_labels))
    print("预测正确数量：", correct_count)
    print("预测错误数量：", len(true_labels) - correct_count)
    print(f"完整测试集准确率：{total_accuracy:.2f}%")

    # 生成20张随机图片网格
    print("\n正在生成20张随机图片的预测结果……")

    selected_indices = visualize_prediction_grid(
        images,
        true_labels,
        predicted_labels
    )

    # 寻找预测正确且置信度低于99.99%的样本
    confidence_values = np.max(all_probabilities, axis=1)

    correct_indices = np.where(
        (predicted_labels == true_labels) &
        (confidence_values < 0.9999)
    )[0]

    if len(correct_indices) > 0:

        # 选择置信度最接近90%的正确样本
        probability_index = int(
            correct_indices[
                np.argmin(
                    np.abs(
                        confidence_values[correct_indices] - 0.90
                    )
                )
            ]
        )

        print("\n========== 预测正确案例 ==========")
        print("测试集图片编号：", probability_index)
        print("真实类别：", CLASS_NAMES[int(true_labels[probability_index])])
        print("预测类别：", CLASS_NAMES[int(predicted_labels[probability_index])])
        print(
            "预测置信度：",
            f"{confidence_values[probability_index] * 100:.2f}%"
        )

        visualize_prediction_probability(
            images[probability_index],
            int(true_labels[probability_index]),
            int(predicted_labels[probability_index]),
            all_probabilities[probability_index]
        )

    else:
        print("没有找到置信度低于99.99%的正确样本。")

    # 查找一张预测错误的图片
    error_indices = np.where(
        predicted_labels != true_labels
    )[0]

    if len(error_indices) > 0:

        rng = np.random.default_rng(42)

        error_index = int(rng.choice(error_indices))

        error_image = images[error_index]

        error_true_label = int(true_labels[error_index])
        error_predicted_label = int(predicted_labels[error_index])

        error_probabilities = all_probabilities[error_index]

        print("\n========== 错误案例分析 ==========")
        print("测试集中的图片编号：", error_index)
        print("真实类别：", CLASS_NAMES[error_true_label])
        print("预测类别：", CLASS_NAMES[error_predicted_label])

        print(
            "预测置信度：",
            f"{error_probabilities[error_predicted_label] * 100:.2f}%"
        )

        visualize_prediction_error(
            error_image,
            error_true_label,
            error_predicted_label,
            error_probabilities
        )

    else:

        print("\n完整测试集中没有发现预测错误的图片。")

    print("\n========== 全部预测任务完成 ==========")
    print("图片保存目录：", RESULTS_DIR)
