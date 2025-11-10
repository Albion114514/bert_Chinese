# predict.py
import torch
import joblib  # 用来加载保存的label_encoder
import os
# 导入 Bert的分词器和分类模型
from transformers import BertTokenizerFast, BertForSequenceClassification
import json  # 读配置文件用的，里面有各种路径和参数

# 先读配置文件，所有参数都存在这里，别硬编码！我之前硬编码改死我了
with open('config.json', 'r', encoding='utf-8') as file:
    config = json.load(file)
best_dir = config["BEST_MODEL_PATH"]  # 最好的模型存在这里
max_len = config["MAX_LENGTH"]  # 最大长度，和训练时保持一致！
num_cls = config["NUM_CLASSES"]  # 类别数量
model_path_bin = config["MODEL_PATH_BIN"]  # 模型bin文件路径


class TextClassifier:
    def __init__(self, model_path, device=None):
        """
        初始化文本分类器，加载模型和分词器

        参数：
        - model_path: 训练好的模型文件夹路径（里面要有权重文件和配置）
        - device: 用CPU还是GPU，默认自动选（有GPU就用GPU，快很多）

        注意事项：
        1. 模型路径一定要对，不然找不到文件
        2. 分词器要和训练时用的一样，不然结果会错
        3. 权重文件可能是pytorch_model.bin或者model.safetensors，这里都处理了
        """
        # 选设备，优先GPU，没有就用CPU
        self.device = torch.device("cuda" if (torch.cuda.is_available() and device != "cpu") else "cpu")
        print(f"使用设备: {self.device}")  # 打印一下用的啥设备，心里有数

        # 保存模型目录，后面加载文件都从这里找
        self.model_dir = model_path

        # 从配置里拿基础模型路径（和训练时一样！）、最大长度、类别数
        self.base_model_name = config["MODEL_PATH"]
        self.max_length = config["MAX_LENGTH"]
        num_labels = config["NUM_CLASSES"]

        try:
            # 1) 加载分词器，必须和训练时用的一样！不然分词结果不对
            self.tokenizer = BertTokenizerFast.from_pretrained(self.base_model_name)

            # 2) 加载模型结构，先加载基础的BERT，再加载我们训练好的权重
            self.model = BertForSequenceClassification.from_pretrained(
                self.base_model_name,
                num_labels=num_labels  # 类别数量要对上
            )

            # 2.1) 加载训练好的权重，可能有两种格式
            wt_bin = os.path.join(self.model_dir, "pytorch_model.bin")  # 第一种格式
            wt_safe = os.path.join(self.model_dir, "model.safetensors")  # 第二种格式

            if os.path.exists(wt_safe):  # 先看有没有safetensors格式
                from safetensors.torch import load_file
                state_dict = load_file(wt_safe)
                self.model.load_state_dict(state_dict)  # 加载权重
            elif os.path.exists(wt_bin):  # 再看有没有bin格式
                # 加载的时候指定设备，避免GPU内存问题
                self.model.load_state_dict(torch.load(wt_bin, map_location=self.device))
            else:  # 都没有就报错
                raise FileNotFoundError(f"没找到权重文件！{wt_bin} 或 {wt_safe} 都没有")

            # 把模型放到选好的设备上
            self.model.to(self.device)
            # 设为评估模式，关闭dropout等，预测更稳定
            self.model.eval()

            # 3) 加载标签编码器，用来把数字标签转回文字
            self.label_encoder_path = os.path.join(self.model_dir, "label_encoder.pkl")
            self.label_encoder = joblib.load(self.label_encoder_path)  # 用joblib加载

            print(f"模型加载成功（路径：{self.model_dir}）")

        except FileNotFoundError as e:
            # 文件找不到的错误，提示清楚点
            raise FileNotFoundError(f"模型文件少了：{e.filename}，检查路径对不对！")
        except Exception as e:
            # 其他错误，比如模型结构不对
            raise RuntimeError(f"模型初始化失败：{str(e)}，可能是配置错了？")

    def predict(self, text):
        """
        预测单个文本的类别

        参数：
        - text: 要预测的文本字符串

        返回：
        - 预测的标签（文字，比如“体育”）
        - 置信度（0-1之间，越大越确定）

        步骤：
        1. 用分词器把文本转成模型能懂的格式
        2. 把数据移到和模型同一个设备上
        3. 关闭梯度计算（预测不用训练，省内存）
        4. 模型输出logits，转成概率（softmax）
        5. 找概率最大的那个类别，转成文字标签
        """
        # 1. 编码文本，参数要和训练时一样！
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,  # 和训练时的MAX_LENGTH一致！
            padding=True,  # 补全
            truncation=True,  # 截断
            return_tensors="pt"  # 返回tensor
        )

        # 2. 移到设备上（和模型同一个设备）
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 3. 预测，不计算梯度
        with torch.no_grad():
            outputs = self.model(** inputs)  # 喂给模型
            # 把logits转成概率（0-1之间）
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # 找最大概率的索引和值
            confidence, predicted_class = torch.max(predictions, dim=1)

            # 把数字类别转回文字标签，注意要先转成cpu的numpy格式
            predicted_label = self.label_encoder.inverse_transform([predicted_class.cpu().item()])[0]
            # 置信度转成浮点数
            confidence_score = confidence.cpu().item()

        return predicted_label, confidence_score


def test_trained_model():
    """
    测试训练好的模型好不好用，跑几个例子看看结果

    注意：
    - 模型路径从config里拿，别硬写死，不然换个模型又要改代码
    - 测试文本最好覆盖所有类别，看看每个类别的预测准不准
    """
    # 从配置里拿模型路径
    model_path = config["BEST_MODEL_PATH"]  # 比如：r"C:\xxx\best_model"

    try:
        # 初始化分类器
        classifier = TextClassifier(model_path)
    except Exception as e:
        print(f"初始化分类器失败：{str(e)}")
        return

    # 准备几个测试文本，涵盖不同类别，记得写预期结果方便核对
    test_texts = [
        # 预期：体育
        "科比27分湖人止连败回西部第2 3巨头轮休马刺败北新浪体育讯北京时间4月13日，西部巨头狭路相逢，湖人主场以102-93击败全替补出场的马刺，将最终对阵的悬念留到最后一场。",
        # 预期：财经
        "央行宣布下调存款准备金率0.5个百分点，释放长期资金约1万亿元，这是今年首次全面降准。",
        # 预期：娱乐
        "这部电影的演员表演出色，剧情扣人心弦，获得了观众和影评人的一致好评。",
        # 新增测试：科技
        "华为发布全新Mate 70系列手机，搭载自研麒麟9010芯片，支持5.5G网络和卫星通话功能。"
    ]

    print("\n模型预测测试:")
    print("=" * 60)

    # 循环每个测试文本，打印结果
    for i, text in enumerate(test_texts, 1):
        label, confidence = classifier.predict(text)
        print(f"样例 {i}:")
        print(f"文本: {text[:60]}...")  # 太长了截断显示，看前60个字符
        print(f"预测类别: {label}")
        print(f"置信度: {confidence:.2%}")  # 转成百分比，看得清楚
        print("-" * 60)


# 运行这个脚本就会执行测试
if __name__ == "__main__":
    test_trained_model()