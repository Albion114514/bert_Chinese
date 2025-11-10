# data_loader.py
import os
import torch
# 从torch的工具里导入Dataset和DataLoader，这俩是处理数据用的
from torch.utils.data import Dataset, DataLoader
# 标签编码器，把文字标签转成数字的，模型只认数字
from sklearn.preprocessing import LabelEncoder
# 用来分割训练集和验证集的
from sklearn.model_selection import train_test_split


class TextClassificationDataset(Dataset):
    """
    这个是文本分类专用的数据集类，必须继承Dataset才行
    里面主要存两个东西：encodings（分词后的结果）和labels（标签）
    后面用DataLoader加载数据的时候会用到这个类
    """

    def __init__(self, encodings, labels):
        # 初始化的时候把这俩存起来
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        # 返回数据有多少条，直接用标签的长度就行，因为一一对应
        return len(self.labels)

    def __getitem__(self, idx):
        # 按索引取数据，必须实现这个方法不然用不了
        # 把encodings里的每个key对应的value转成tensor，不然模型不认
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # 标签也要转成tensor，存在'labels'这个key里，模型会找这个key
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def load_thucnews_data(file_path):
    """
    读取THUCNews格式的数据集，格式必须是“标签\t文本内容”这种！
    比如一行是“体育\t科比27分湖人止连败...”，中间是Tab隔开
    我之前因为格式错了卡了好久，一定要注意！

    步骤：
    1. 打开文件，一行一行读
    2. 跳过空行，不然会报错
    3. 按第一个Tab分割成标签和文本，分割不了的行就跳过（会打印提示）
    4. 过滤掉空文本或者太短的（少于10个字符），不然训练效果差
    5. 最后返回文本列表和标签列表
    """
    texts = []  # 存所有文本
    labels = []  # 存所有标签

    try:
        # 用utf-8编码打开，errors='ignore'是为了避免有些奇怪的字符报错
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # 枚举行数，从1开始，方便报错时知道哪行有问题
            for line_num, line in enumerate(f, 1):
                line = line.strip()  # 去掉前后的空格和换行
                if not line:  # 空行就跳过
                    continue

                # 按第一个Tab分割，最多分两部分（标签和文本）
                parts = line.split('\t', 1)
                if len(parts) != 2:  # 没分成两部分就是格式错了
                    print(f"跳过格式错误的行 {line_num}: {line[:50]}...")  # 只显示前50个字符
                    continue

                label, text = parts
                text = text.strip()  # 文本再清一下空格

                # 文本不为空且至少10个字符才要，太短的没意义
                if text and len(text) >= 10:
                    texts.append(text)
                    labels.append(label.strip())  # 标签也清一下空格

        print(f"成功加载 {len(texts)} 条数据")
        print(f"类别分布: {set(labels)}")  # 看看有哪些类别

    except Exception as e:
        print(f"读取文件失败: {e}")  # 比如文件路径错了就会到这
        return [], []  # 出错了就返回空列表

    return texts, labels


def preprocess_and_split_data(texts, labels, tokenizer, max_length=128, test_size=0.2):
    """
    数据预处理+分割训练集和验证集的函数，一步到位

    做了这几件事：
    1. 用LabelEncoder把文字标签转成数字（比如“体育”→0，“财经”→1）
    2. 用train_test_split把数据分成训练集（80%）和验证集（20%），test_size是验证集比例
    3. 用tokenizer对文本分词，转成模型能懂的格式（input_ids、attention_mask这些）
       - max_length是最大长度，超过会截断，不够会补全
       - return_tensors='pt'表示返回pytorch的tensor格式

    返回的东西：
    - 训练集的encodings和labels
    - 验证集的encodings和labels
    - 类别数量（num_classes）
    - 标签编码器（后面预测时要用来转回去）
    """
    # 初始化标签编码器
    label_encoder = LabelEncoder()
    # 把labels转成数字，存成label_ids
    label_ids = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)  # 类别数量

    print(f"标签编码完成，共 {num_classes} 个类别: {list(label_encoder.classes_)}")

    # 分割训练集和验证集
    # stratify=label_ids是为了让训练集和验证集的类别分布差不多
    # random_state=42是固定随机种子，这样每次分割结果一样，方便调试
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, label_ids,
        test_size=test_size,
        random_state=42,
        stratify=label_ids
    )

    print(f"训练集: {len(train_texts)} 条")
    print(f"验证集: {len(val_texts)} 条")

    print("开始分词处理...")  # 分词可能有点慢，等一下

    # 处理训练集文本
    train_encodings = tokenizer(
        train_texts,
        truncation=True,  # 超过max_length就截断
        padding=True,  # 不够max_length就补全
        max_length=max_length,
        return_tensors='pt'  # 返回pytorch的tensor
    )

    # 处理验证集文本，参数和训练集一样，不然会出错！
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )

    return (train_encodings, train_labels), (val_encodings, val_labels), num_classes, label_encoder


def create_data_loaders(data_path, tokenizer, max_length=128, batch_size=8, test_size=0.2):
    """
    一站式创建数据加载器的函数，给训练用的

    步骤：
    1. 先调用load_thucnews_data加载原始数据
    2. 调用preprocess_and_split_data预处理和分割
    3. 用TextClassificationDataset创建数据集
    4. 用DataLoader把数据集转成批次（batch），方便训练时一批一批喂给模型

    参数里的batch_size是每个批次有多少条数据，太大可能内存不够，太小训练慢
    num_workers=0是因为Windows系统设成别的可能会报错，先用0试试
    """
    # 加载原始数据
    texts, labels = load_thucnews_data(data_path)
    if not texts:  # 如果没加载到数据，就报错
        raise ValueError("数据加载失败，请检查文件路径和格式")

    # 预处理和划分
    (train_encodings, train_labels), (
        val_encodings, val_labels), num_classes, label_encoder = preprocess_and_split_data(
        texts, labels, tokenizer, max_length, test_size
    )

    # 创建数据集实例
    train_dataset = TextClassificationDataset(train_encodings, train_labels)
    val_dataset = TextClassificationDataset(val_encodings, val_labels)

    # 创建训练集加载器，shuffle=True表示每个epoch都打乱顺序
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Windows用户别改这个，可能会炸
    )

    # 验证集不用打乱，所以shuffle=False
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, num_classes, label_encoder