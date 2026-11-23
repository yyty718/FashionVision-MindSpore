
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. 项目路径
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_PATH = RESULTS_DIR / "evaluation_results.json"


# =========================
# 2. 读取真实评估结果
# =========================

if not EVALUATION_PATH.exists():
    raise FileNotFoundError(
        f"找不到评估结果：{EVALUATION_PATH}\n"
        "请先运行：python src/evaluate.py"
    )

with EVALUATION_PATH.open("r", encoding="utf-8") as file:
    evaluation = json.load(file)

MODEL_ORDER = [
    "mlp",
    "cnn",
    "improved_cnn",
    "improved_cnn_no_dropout",
]

missing_models = [
    model_key
    for model_key in MODEL_ORDER
    if model_key not in evaluation["models"]
]

if missing_models:
    raise ValueError(
        "评估结果缺少以下模型，请重新运行 python src/evaluate.py："
        + ", ".join(missing_models)
    )

model_results = evaluation["models"]
full_model_names = [
    model_results[model_key]["display_name"]
    for model_key in MODEL_ORDER
]
model_names = [
    name.replace(" Without Dropout", "\nWithout Dropout")
    for name in full_model_names
]

accuracy = [
    model_results[model_key]["accuracy"] * 100
    for model_key in MODEL_ORDER
]
macro_f1 = [
    model_results[model_key]["macro_f1"] * 100
    for model_key in MODEL_ORDER
]

# Fashion-MNIST 类别名称
class_names = evaluation["class_names"]

# 每个模型在各类别上的 F1-score
class_f1 = {
    full_model_names[index]: [
        item["f1"]
        for item in model_results[model_key]["class_metrics"]
    ]
    for index, model_key in enumerate(MODEL_ORDER)
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
    model_results[model_key]["display_name"]: np.array(
        model_results[model_key]["confusion_matrix"]
    )
    for model_key in MODEL_ORDER
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
print("评估结果来源：", EVALUATION_PATH)
print("评估生成时间：", evaluation["generated_at"])
print("图表保存位置：", RESULTS_DIR)
print("生成的文件：")
print("- accuracy_comparison.png")
print("- macro_f1_comparison.png")
print("- class_f1_comparison.png")
print("- confusion_matrix_comparison.png")
