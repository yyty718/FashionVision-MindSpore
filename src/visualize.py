
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. 项目路径
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 实验结果
# =========================

model_names = [
    "MLP",
    "CNN",
    "Improved CNN",
    "Improved CNN\nWithout Dropout"
]

# Accuracy（准确率）
accuracy = [88.18, 91.07, 90.88, 91.40]

# Macro F1-score
macro_f1 = [88.22, 90.97, 90.90, 91.49]

# Fashion-MNIST 类别名称
class_names = [
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

# 每个模型在各类别上的 F1-score
class_f1 = {
    "MLP": [
        0.8289, 0.9755, 0.8026, 0.8883, 0.8090,
        0.9535, 0.7041, 0.9347, 0.9722, 0.9531
    ],
    "CNN": [
        0.8657, 0.9869, 0.8557, 0.9140, 0.8498,
        0.9821, 0.7344, 0.9604, 0.9786, 0.9694
    ],
    "Improved CNN": [
        0.8614, 0.9879, 0.8693, 0.9064, 0.8531,
        0.9844, 0.7243, 0.9574, 0.9826, 0.9636
    ],
    "Improved CNN Without Dropout": [
        0.8434, 0.9900, 0.8803, 0.9115, 0.8590,
        0.9855, 0.7561, 0.9681, 0.9839, 0.9715
    ]
}


# =========================
# 3. Matplotlib 中文显示设置
# =========================

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS"
]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 4. 绘制 Accuracy 对比图
# =========================

plt.figure(figsize=(9, 6))

bars = plt.bar(model_names, accuracy)

plt.title("不同模型的 Accuracy 对比", fontsize=15)
plt.xlabel("模型")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bar, value in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.15,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig(RESULTS_DIR / "accuracy_comparison.png", dpi=300)
plt.close()


# =========================
# 5. 绘制 Macro F1 对比图
# =========================

plt.figure(figsize=(9, 6))

bars = plt.bar(model_names, macro_f1)

plt.title("不同模型的 Macro F1-score 对比", fontsize=15)
plt.xlabel("模型")
plt.ylabel("Macro F1-score (%)")
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bar, value in zip(bars, macro_f1):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.15,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig(RESULTS_DIR / "macro_f1_comparison.png", dpi=300)
plt.close()


# =========================
# 6. 绘制各类别 F1-score 对比图
# =========================

plt.figure(figsize=(13, 7))

for model_name, scores in class_f1.items():
    plt.plot(
        class_names,
        scores,
        marker="o",
        linewidth=2,
        label=model_name
    )

plt.title("不同模型在各类别上的 F1-score 对比", fontsize=15)
plt.xlabel("Fashion-MNIST 类别")
plt.ylabel("F1-score")
plt.ylim(0.65, 1.02)
plt.xticks(rotation=25)
plt.grid(linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()

plt.savefig(RESULTS_DIR / "class_f1_comparison.png", dpi=300)
plt.close()



# =========================
# 7. 绘制四个模型的混淆矩阵
# =========================

confusion_matrices = {
    "MLP": np.array([
        [797, 1, 15, 30, 2, 2, 150, 1, 2, 0],
        [4, 957, 3, 28, 3, 0, 3, 0, 2, 0],
        [16, 1, 813, 18, 93, 0, 58, 0, 1, 0],
        [12, 2, 8, 915, 35, 1, 24, 0, 3, 0],
        [0, 1, 89, 28, 811, 0, 70, 0, 1, 0],
        [0, 0, 0, 1, 0, 975, 0, 16, 0, 8],
        [91, 0, 95, 33, 58, 0, 715, 0, 8, 0],
        [0, 0, 0, 0, 0, 43, 0, 938, 1, 18],
        [3, 0, 3, 6, 3, 6, 10, 6, 963, 0],
        [0, 0, 0, 1, 0, 18, 1, 46, 0, 934]
    ]),

    "CNN": np.array([
        [854, 0, 21, 17, 4, 1, 94, 0, 9, 0],
        [0, 978, 0, 18, 2, 0, 0, 0, 2, 0],
        [12, 1, 907, 7, 34, 0, 37, 0, 2, 0],
        [9, 1, 11, 941, 19, 0, 17, 0, 2, 0],
        [0, 0, 92, 33, 846, 1, 27, 0, 1, 0],
        [0, 0, 0, 0, 0, 990, 0, 7, 0, 3],
        [94, 1, 88, 39, 85, 0, 683, 0, 10, 0],
        [0, 0, 0, 0, 0, 17, 0, 958, 0, 25],
        [3, 1, 1, 4, 1, 2, 2, 3, 983, 0],
        [1, 0, 0, 0, 0, 5, 0, 27, 0, 967]
    ]),

    "Improved CNN": np.array([
        [873, 0, 7, 12, 3, 1, 98, 0, 6, 0],
        [0, 981, 0, 13, 2, 0, 2, 0, 2, 0],
        [9, 1, 841, 9, 63, 0, 76, 0, 1, 0],
        [23, 3, 7, 901, 22, 0, 43, 0, 1, 0],
        [0, 0, 40, 27, 871, 0, 62, 0, 0, 0],
        [0, 0, 0, 0, 0, 976, 0, 20, 0, 4],
        [119, 0, 40, 22, 79, 0, 729, 0, 11, 0],
        [0, 0, 0, 0, 0, 2, 0, 977, 0, 21],
        [3, 1, 0, 4, 2, 1, 2, 1, 986, 0],
        [0, 0, 0, 0, 0, 3, 1, 43, 0, 953]
    ]),

    "Improved CNN Without Dropout": np.array([
        [765, 0, 12, 36, 2, 1, 180, 0, 4, 0],
        [0, 986, 0, 10, 1, 0, 2, 0, 1, 0],
        [9, 1, 846, 12, 56, 0, 75, 0, 1, 0],
        [2, 3, 5, 953, 12, 0, 24, 0, 1, 0],
        [0, 0, 24, 38, 844, 0, 93, 0, 1, 0],
        [0, 0, 0, 0, 0, 985, 0, 14, 0, 1],
        [36, 1, 35, 38, 47, 0, 840, 0, 3, 0],
        [0, 0, 0, 0, 0, 3, 0, 986, 0, 11],
        [2, 1, 0, 4, 3, 2, 7, 2, 979, 0],
        [0, 0, 0, 0, 0, 8, 1, 35, 0, 956]
    ])
}

fig, axes = plt.subplots(2, 2, figsize=(18, 15))

for ax, (model_name, matrix) in zip(axes.flat, confusion_matrices.items()):

    image = ax.imshow(matrix, cmap="Blues")

    ax.set_title(model_name, fontsize=14)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # 在每个单元格中标注数量
    threshold = matrix.max() / 2

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=7
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("四种模型的混淆矩阵对比", fontsize=18)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "confusion_matrix_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================
# 8. 完成提示
# =========================

print("实验结果可视化完成！")
print("图表保存位置：", RESULTS_DIR)
print("生成的文件：")
print("- accuracy_comparison.png")
print("- macro_f1_comparison.png")
print("- class_f1_comparison.png")
print("- confusion_matrix_comparison.png")