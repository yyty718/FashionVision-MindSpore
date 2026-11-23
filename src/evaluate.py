
"""
FashionVision-MindSpore
统一模型评估入口
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import mindspore as ms

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)

from dataset import create_dataset
from train import MODEL_REGISTRY


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

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


def resolve_project_path(path):
    """将相对路径解析为项目根目录下的路径。"""
    resolved_path = Path(path).expanduser()

    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path

    return resolved_path


def load_training_record(checkpoint_dir, model_key):
    """读取训练时生成的参数记录；旧检查点没有记录时返回空字典。"""
    record_path = checkpoint_dir / f"{model_key}_training.json"

    if not record_path.exists():
        return {}

    with record_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_model(model_key, checkpoint_dir, batch_size):
    """评估一个指定模型，返回该模型的全部测试指标。"""

    model_config = MODEL_REGISTRY[model_key]
    display_name = model_config["display_name"]
    checkpoint_path = (
        checkpoint_dir / model_config["checkpoint_name"]
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"找不到 {display_name} 的检查点：{checkpoint_path}"
        )

    network = model_config["model_class"]()
    parameter_dict = ms.load_checkpoint(str(checkpoint_path))
    parameters_not_loaded, parameters_not_used = ms.load_param_into_net(
        network,
        parameter_dict
    )

    if parameters_not_loaded:
        raise RuntimeError(
            f"{display_name} 有参数未加载：{parameters_not_loaded}"
        )

    if parameters_not_used:
        print(
            f"警告：{display_name} 的检查点中有参数未被网络使用：",
            parameters_not_used
        )

    network.set_train(False)

    test_dataset = create_dataset(
        usage="test",
        shuffle=False,
        batch_size=batch_size
    )

    all_labels = []
    all_predictions = []

    for batch in test_dataset.create_dict_iterator():
        images = batch["image"]
        labels = batch["label"]
        outputs = network(images)
        predictions = outputs.asnumpy().argmax(axis=1)

        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.asnumpy().tolist())

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(10)),
        average="macro",
        zero_division=0
    )
    precision, recall, class_f1, support = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=list(range(10)),
            zero_division=0
        )
    )
    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(10))
    )

    print("=" * 70)
    print(f"{display_name} 测试结果")
    print("=" * 70)
    print("检查点：", checkpoint_path)
    print("测试集批次数量：", test_dataset.get_dataset_size())
    print("测试样本数量：", len(all_labels))

    print(f"\nAccuracy（准确率）：{accuracy:.4f}")
    print(f"Accuracy（百分比）：{accuracy * 100:.2f}%")
    print(f"Macro F1-score：{macro_f1:.4f}")

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

    print("混淆矩阵（Confusion Matrix）：")
    print(matrix)

    class_metrics = []
    for index, class_name in enumerate(CLASS_NAMES):
        class_metrics.append({
            "class_index": index,
            "class_name": class_name,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(class_f1[index]),
            "support": int(support[index]),
        })

    return {
        "model_key": model_key,
        "display_name": display_name,
        "checkpoint_path": str(checkpoint_path),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "class_metrics": class_metrics,
        "confusion_matrix": matrix.tolist(),
        "training_record": load_training_record(
            checkpoint_dir,
            model_key
        ),
    }


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="评估 Fashion-MNIST MLP/CNN 对比模型"
    )
    parser.add_argument(
        "--model",
        choices=[*MODEL_REGISTRY, "all"],
        default="all",
        help="要评估的模型；默认 all，评估全部四种模型"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="检查点目录，默认项目根目录下的 checkpoints"
    )
    parser.add_argument(
        "--output",
        default="results/evaluation_results.json",
        help="评估结果 JSON 保存路径"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="测试集批大小，默认 64"
    )
    parser.add_argument(
        "--device-target",
        default="CPU",
        help="MindSpore 运行设备，默认 CPU"
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size 必须大于 0")

    return args


def main():
    """评估一个或全部模型并保存结果。"""
    args = parse_args()

    ms.set_context(
        mode=ms.PYNATIVE_MODE,
        device_target=args.device_target
    )

    if args.model == "all":
        model_keys = list(MODEL_REGISTRY)
    else:
        model_keys = [args.model]

    checkpoint_dir = resolve_project_path(args.checkpoint_dir)
    output_path = resolve_project_path(args.output)
    model_results = {}

    for model_key in model_keys:
        model_results[model_key] = evaluate_model(
            model_key=model_key,
            checkpoint_dir=checkpoint_dir,
            batch_size=args.batch_size
        )

    output = {
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "dataset": "Fashion-MNIST test",
        "device_target": args.device_target,
        "checkpoint_dir": str(checkpoint_dir),
        "class_names": CLASS_NAMES,
        "models": model_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("评估任务完成！")
    print("评估结果：", output_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
