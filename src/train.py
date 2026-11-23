
"""
FashionVision-MindSpore
统一模型训练入口
"""

import argparse
import json
import random
import time
from pathlib import Path

import mindspore as ms
import mindspore.nn as nn
import numpy as np
from mindspore.train import Model
from mindspore.train.callback import LossMonitor

from dataset import create_dataset
from models import MLP, CNN, ImprovedCNN, ImprovedCNNWithoutDropout


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 可用模型及其检查点文件
MODEL_REGISTRY = {
    "mlp": {
        "display_name": "MLP",
        "model_class": MLP,
        "checkpoint_name": "mlp.ckpt",
    },
    "cnn": {
        "display_name": "CNN",
        "model_class": CNN,
        "checkpoint_name": "cnn.ckpt",
    },
    "improved_cnn": {
        "display_name": "Improved CNN",
        "model_class": ImprovedCNN,
        "checkpoint_name": "improved_cnn.ckpt",
    },
    "improved_cnn_no_dropout": {
        "display_name": "Improved CNN Without Dropout",
        "model_class": ImprovedCNNWithoutDropout,
        "checkpoint_name": "improved_cnn_no_dropout.ckpt",
    },
}


def set_random_seed(seed):
    """统一设置 Python、NumPy 和 MindSpore 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


def get_checkpoint_dir(path):
    """解析检查点目录；相对路径以项目根目录为基准。"""
    checkpoint_dir = Path(path).expanduser()

    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir

    return checkpoint_dir


def train_model(
    model_key,
    epochs,
    batch_size,
    learning_rate,
    device_target,
    seed,
    checkpoint_dir,
    per_print_times
):
    """训练一个指定模型并保存检查点。"""

    if model_key not in MODEL_REGISTRY:
        available_models = ", ".join(MODEL_REGISTRY)
        raise ValueError(
            f"未知模型：{model_key}。可选模型：{available_models}"
        )

    model_config = MODEL_REGISTRY[model_key]
    display_name = model_config["display_name"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 必须在创建网络之前设置随机种子，保证模型初始化可复现
    set_random_seed(seed)
    network = model_config["model_class"]()

    # 再次重置种子，使不同模型使用相同的训练集打乱顺序
    set_random_seed(seed)
    train_dataset = create_dataset(
        usage="train",
        shuffle=True,
        batch_size=batch_size
    )

    dataset_size = train_dataset.get_dataset_size()
    parameter_count = sum(
        parameter.size for parameter in network.trainable_params()
    )

    loss_fn = nn.CrossEntropyLoss()

    optimizer = nn.Adam(
        params=network.trainable_params(),
        learning_rate=learning_rate
    )

    model = Model(
        network=network,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics={"Accuracy": nn.Accuracy()}
    )

    loss_monitor = LossMonitor(per_print_times=per_print_times)

    print("=" * 70)
    print(f"开始训练：{display_name}")
    print(f"模型标识：{model_key}")
    print(f"训练集批次数量：{dataset_size}")
    print(f"参数量：{parameter_count:,}")
    print(
        f"Epochs：{epochs}  Batch Size：{batch_size}  "
        f"Learning Rate：{learning_rate}  Device：{device_target}"
    )
    print(f"随机种子：{seed}")
    print("=" * 70)

    start_time = time.perf_counter()
    model.train(
        epochs,
        train_dataset,
        callbacks=[loss_monitor],
        dataset_sink_mode=False
    )
    elapsed_seconds = time.perf_counter() - start_time

    checkpoint_path = checkpoint_dir / model_config["checkpoint_name"]

    ms.save_checkpoint(
        network,
        str(checkpoint_path)
    )

    # 保存本次训练的配置信息，方便之后写报告和复现实验
    record_path = checkpoint_dir / f"{model_key}_training.json"
    training_record = {
        "model": display_name,
        "model_key": model_key,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
        "device_target": device_target,
        "seed": seed,
        "parameter_count": parameter_count,
        "train_dataset_batches": dataset_size,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "checkpoint_path": str(checkpoint_path),
    }

    with record_path.open("w", encoding="utf-8") as file:
        json.dump(training_record, file, ensure_ascii=False, indent=2)

    print(f"\n{display_name} 模型训练完成！")
    print("保存位置：", checkpoint_path)
    print("训练记录：", record_path)
    print(f"训练耗时：{elapsed_seconds:.2f} 秒")

    return checkpoint_path


def parse_args():
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="训练 Fashion-MNIST MLP/CNN 对比模型"
    )
    parser.add_argument(
        "--model",
        choices=[*MODEL_REGISTRY, "all"],
        default="all",
        help="要训练的模型；默认 all，依次训练全部四种模型"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="训练轮数，默认 5"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="批大小，默认 64"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Adam 学习率，默认 0.001"
    )
    parser.add_argument(
        "--device-target",
        default="CPU",
        help="MindSpore 运行设备，例如 CPU、GPU 或 Ascend，默认 CPU"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="检查点保存目录，默认项目根目录下的 checkpoints"
    )
    parser.add_argument(
        "--per-print-times",
        type=int,
        default=100,
        help="每训练多少个 step 输出一次损失，默认 100"
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs 必须大于 0")

    if args.batch_size <= 0:
        parser.error("--batch-size 必须大于 0")

    if args.learning_rate <= 0:
        parser.error("--learning-rate 必须大于 0")

    if args.per_print_times <= 0:
        parser.error("--per-print-times 必须大于 0")

    return args


def main():
    """配置运行环境并训练一个或全部模型。"""

    args = parse_args()

    ms.set_context(
        mode=ms.PYNATIVE_MODE,
        device_target=args.device_target
    )

    if args.model == "all":
        model_keys = list(MODEL_REGISTRY)
    else:
        model_keys = [args.model]

    checkpoint_dir = get_checkpoint_dir(args.checkpoint_dir)
    saved_checkpoints = []

    for model_key in model_keys:
        checkpoint_path = train_model(
            model_key=model_key,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device_target=args.device_target,
            seed=args.seed,
            checkpoint_dir=checkpoint_dir,
            per_print_times=args.per_print_times
        )
        saved_checkpoints.append(checkpoint_path)

    print("\n" + "=" * 70)
    print(f"训练任务全部完成，共训练 {len(saved_checkpoints)} 个模型。")
    for checkpoint_path in saved_checkpoints:
        print("-", checkpoint_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
