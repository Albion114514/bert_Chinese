import torch
import os
import json  # 这里导入json模块，是后来加上的，用来读取配置文件的，不然配置参数不好管理
from transformers import BertTokenizerFast, BertForSequenceClassification
from data_loader import create_data_loaders  # 从data_loader文件里导入创建数据加载器的函数
from model_trainer import train_bert_model  # 从model_trainer导入训练模型的函数


# 这个函数是用来加载配置文件的，配置文件里有各种路径和参数，比如模型放哪、数据放哪
def load_config(config_path="config.json"):
    """加载JSON配置文件，要是配置文件找不到就会报错，得记着配置文件名叫config.json"""
    # 先检查配置文件在不在指定路径
    if not os.path.exists(config_path):
        # 找不到就抛出错误，告诉我们哪个文件没找到
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    # 打开文件，用utf-8编码读，然后返回读到的内容（是个字典）
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_environment(config_data):  # 这个函数用来设置训练的环境，需要传配置字典进来
    """设置训练环境，主要是选设备（CPU/GPU）和检查模型文件全不全"""
    # 先看看电脑有没有GPU，有就用cuda，没有就只能用cpu了，打印出来看看用的啥
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # BERT模型必须要有这几个文件才能用，记一下这几个文件名，少一个都不行
    required_files = ['config.json', 'pytorch_model.bin', 'vocab.txt']
    for file in required_files:
        # 拼接文件路径，MODEL_PATH是配置里写的模型存放路径
        file_path = f"{config_data['MODEL_PATH']}/{file}"
        # 检查每个文件在不在
        if not os.path.exists(file_path):
            # 少了文件就报错，告诉我们缺了哪个
            raise FileNotFoundError(f"缺少BERT模型文件: {file_path}")

    # 把选好的设备返回出去，后面模型和数据都要用
    return device


def main():
    """主训练函数，整个训练流程都在这里面，一步一步来"""
    print("=" * 60)
    print("BERT 文本分类训练 - THUCNews 数据集")  # 这次是用THUCNews数据集做文本分类
    print("=" * 60)

    try:
        # 先加载配置文件里的所有参数，后面训练要用的路径、超参数都从这里拿
        config_data = load_config()

        # 1. 先设置好环境，主要是确定用CPU还是GPU，还有检查模型文件
        device = setup_environment(config_data)  # 把配置传进去

        # 2. 加载BERT的分词器，分词器是把文本转成模型能看懂的数字的
        print("\n1. 加载BERT分词器...")
        tokenizer = BertTokenizerFast.from_pretrained(
            config_data['MODEL_PATH'],  # 分词器的路径就是模型存放的路径，里面有vocab.txt
            do_lower_case=False  # 这里设成False，不把大写转小写，具体为啥以后再想
        )

        # 3. 创建数据加载器，训练和验证的时候要用到，会把数据分成一批一批的
        print("\n2. 创建数据加载器...")
        train_loader, val_loader, num_classes, label_encoder = create_data_loaders(
            data_path=config_data['DATA_PATH'],  # 数据存放的路径，配置里写好了
            tokenizer=tokenizer,  # 刚才加载的分词器
            max_length=config_data['MAX_LENGTH'],  # 句子最长多少个词，配置里定的
            batch_size=config_data['BATCH_SIZE'],  # 每批多少个样本，也是配置里的
            test_size=config_data['TEST_SIZE']  # 多少比例的数据用来做验证，比如0.2就是20%
        )

        # 4. 加载BERT模型，这次是用来做分类的
        print("\n3. 加载BERT模型...")
        model = BertForSequenceClassification.from_pretrained(
            config_data['MODEL_PATH'],  # 模型文件的路径
            num_labels=num_classes  # 要分多少类，前面数据加载器返回的
        )
        model.to(device)  # 把模型放到前面选好的设备上（CPU或GPU）

        # 打印一下模型有多少参数，大概了解下模型大小，numel()是算每个参数的数量
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"分类类别数: {num_classes}")  # 看看要分多少类

        # 5. 开始训练模型啦，调用训练函数
        print("\n4. 开始训练...")
        best_model_path = train_bert_model(
            model=model,  # 刚才加载的模型
            train_loader=train_loader,  # 训练数据加载器
            val_loader=val_loader,  # 验证数据加载器
            label_encoder=label_encoder,  # 标签编码器，可能用来转换标签的
            device=device,  # 设备
            save_path=config_data['SAVE_MODEL_PATH'],  # 模型保存的路径，配置里的
            learning_rate=config_data['LEARNING_RATE'],  # 学习率，调参很重要，配置里定的
            epochs=config_data['EPOCHS']  # 训练多少轮，也是配置里的
        )

        print(f"\n🎊 训练完成！")
        print(f"最佳模型保存在: {best_model_path}")  # 告诉我们最好的模型存在哪了

    # 如果中间有任何错误，就会跑到这里
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")  # 打印错误信息
        raise  # 把错误抛出来，方便看到具体哪里错了


if __name__ == "__main__":
    # 程序入口，先确保保存模型的文件夹存在，没有就创建一个
    config_data = load_config()  # 先加载配置，拿到保存路径
    # exist_ok=True表示如果文件夹已经存在，就不报错
    os.makedirs(config_data['SAVE_MODEL_PATH'], exist_ok=True)
    main()  # 然后执行主函数