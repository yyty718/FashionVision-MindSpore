
"""
FashionVision-MindSpore
数据集加载与预处理模块
"""

import gzip
import shutil
import urllib.request
from pathlib import Path
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import dtype as mstype


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fashion-MNIST 数据存放目录
DATA_DIR = PROJECT_ROOT / "data" / "fashion_mnist"

# Fashion-MNIST 官方数据文件下载地址
BASE_URL = (
    "https://raw.githubusercontent.com/"
    "zalandoresearch/fashion-mnist/master/data/fashion/"
)

# 需要下载的四个文件
DATA_FILES = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]


def download_data():
    """下载并解压 Fashion-MNIST 数据集"""

    # 如果目录不存在，就创建目录
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename in DATA_FILES:

        # 解压后的文件路径
        file_path = DATA_DIR / filename

        # 如果文件已经存在，就不重复下载
        if file_path.exists():
            print(f"文件已存在，跳过：{filename}")
            continue

        # 压缩文件名称
        gz_filename = filename + ".gz"
        gz_path = DATA_DIR / gz_filename

        # 下载地址
        url = BASE_URL + gz_filename

        print(f"正在下载：{gz_filename}")

        try:
            # 下载压缩文件
            urllib.request.urlretrieve(url, gz_path)

            print(f"下载完成，正在解压：{filename}")

            # 解压 gzip 文件
            with gzip.open(gz_path, "rb") as f_in:
                with open(file_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # 删除压缩包，节省空间
            gz_path.unlink()

            print(f"处理完成：{filename}")

        except Exception:
            # 下载或解压失败时，清理不完整文件
            if gz_path.exists():
                gz_path.unlink()

            if file_path.exists():
                file_path.unlink()

            raise

    print("Fashion-MNIST 数据集准备完成！")


def create_dataset(usage="train", shuffle=True, batch_size=64):
    """
    创建 Fashion-MNIST 数据集

    usage: "train" 表示训练集，"test" 表示测试集
    shuffle: 是否打乱数据
    """

    dataset = ds.FashionMnistDataset(
        dataset_dir=str(DATA_DIR),
        usage=usage,
        shuffle=shuffle
    )

    # 图片预处理
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize(
            mean=[0.2860],
            std=[0.3530]
        ),
        vision.HWC2CHW()
    ]

    # 对图片应用预处理
    dataset = dataset.map(
        operations=image_transforms,
        input_columns="image"
    )

    # 将标签转换为 int32 类型
    dataset = dataset.map(
        operations=ds.transforms.TypeCast(mstype.int32),
        input_columns="label"
    )

    # 将数据划分为一个个批次
    dataset = dataset.batch(
        batch_size=batch_size,
        drop_remainder=False
    )

    return dataset


if __name__ == "__main__":

    # 创建训练集
    train_dataset = create_dataset(
        usage="train",
        shuffle=True,
        batch_size=64
    )

    # 创建测试集
    test_dataset = create_dataset(
        usage="test",
        shuffle=False,
        batch_size=64
    )

    print("训练集批次数量：", train_dataset.get_dataset_size())
    print("测试集批次数量：", test_dataset.get_dataset_size())

    # 读取训练集的第一个批次
    for item in train_dataset.create_dict_iterator(
        num_epochs=1,
        output_numpy=True
    ):
        images = item["image"]
        labels = item["label"]

        print("一个批次的图片形状：", images.shape)
        print("一个批次的标签形状：", labels.shape)
        print("图片数据类型：", images.dtype)
        print("标签数据类型：", labels.dtype)

        break