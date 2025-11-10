import torch
import joblib
import os
import json
from transformers import BertTokenizerFast, BertForSequenceClassification
from datetime import datetime

# 创建必要的文件夹
def create_necessary_dirs():
    """创建输入输出文件夹（如果不存在）"""
    if not os.path.exists("./sort_input"):
        os.makedirs("./sort_input")
        print("✅ 已创建输入文件夹: ./sort_input")
    if not os.path.exists("./sort_output"):
        os.makedirs("./sort_output")
        print("✅ 已创建输出文件夹: ./sort_output")

# 读取配置文件
with open('config.json', 'r', encoding='utf-8') as file:
    config = json.load(file)
best_model_path = config["BEST_MODEL_PATH"]

# ===== 标注与日志配置 =====
ANNOTATIONS_DIR = "./sort_output"
ANNOTATIONS_CSV = os.path.join(ANNOTATIONS_DIR, "annotations.csv")
ANNOTATIONS_JSONL = os.path.join(ANNOTATIONS_DIR, "annotations.jsonl")
ENABLE_HUMAN_LABELING = True
ALLOW_NEW_LABELS = True
STRICT_LABEL_CHECK = False

def _ensure_annotations_dir():
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

def _append_csv(row: dict):
    import csv
    _ensure_annotations_dir()
    headers = ["source", "filename", "pred_label", "confidence", "human_labels", "new_labels", "invalid_labels", "timestamp"]
    write_header = not os.path.exists(ANNOTATIONS_CSV)
    with open(ANNOTATIONS_CSV, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if write_header:
            w.writeheader()
        w.writerow(row)

def _append_jsonl(obj: dict):
    _ensure_annotations_dir()
    with open(ANNOTATIONS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _parse_human_labels(input_str: str, known_classes: list, allow_new=True, strict=False):
    tokens = [x.strip() for x in (input_str or "").split(",") if x.strip()]
    if not tokens:
        return [], [], []
    known_set = set(known_classes or [])
    valid, new, invalid = [], [], []
    for t in tokens:
        if t in known_set:
            valid.append(t)
        else:
            if allow_new and not strict:
                new.append(t)
            else:
                invalid.append(t)
    return valid, new, invalid

def sanitize_text(raw_text, *, max_chars=500, min_chars=10):
    """校验与截断文本"""
    if raw_text is None:
        raise ValueError("空文本")
    text = str(raw_text).replace("\r\n", "\n").strip()
    if not text:
        raise ValueError("空文本")
    if "\x00" in text:
        raise ValueError("疑似二进制文件（包含NUL）")
    bad_count = text.count("�")
    if bad_count and (bad_count / max(1, len(text)) > 0.01):
        print("⚠️ 文本包含乱码字符，请检查编码（UTF-8 推荐）")
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
    if len(text) < min_chars:
        raise ValueError(f"文本过短（少于 {min_chars} 字）")
    return text, truncated


class TextClassifier:
    def __init__(self, model_path, device=None):
        """初始化模型"""
        self.device = torch.device("cuda" if (torch.cuda.is_available() and device != "cpu") else "cpu")
        print(f"🔧 使用设备: {self.device}")

        self.model_dir = model_path
        self.base_model_name = config["MODEL_PATH"]
        self.max_length = config["MAX_LENGTH"]
        self.num_labels = config["NUM_CLASSES"]

        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(self.base_model_name)
            self.model = BertForSequenceClassification.from_pretrained(
                self.base_model_name, num_labels=self.num_labels)
            wt_bin = os.path.join(self.model_dir, "pytorch_model.bin")
            wt_safe = os.path.join(self.model_dir, "model.safetensors")
            if os.path.exists(wt_safe):
                from safetensors.torch import load_file
                self.model.load_state_dict(load_file(wt_safe))
            elif os.path.exists(wt_bin):
                self.model.load_state_dict(torch.load(wt_bin, map_location=self.device))
            else:
                raise FileNotFoundError("未找到模型权重文件！")
            self.model.to(self.device)
            self.model.eval()
            self.label_encoder = joblib.load(os.path.join(self.model_dir, "label_encoder.pkl"))
            print(f"✅ 模型加载成功（路径：{self.model_dir}）")
        except Exception as e:
            raise RuntimeError(f"模型初始化失败：{e}")

    def predict(self, text):
        """预测单文本类别"""
        inputs = self.tokenizer(
            text, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            conf, idx = torch.max(probs, dim=1)
            label = self.label_encoder.inverse_transform([idx.cpu().item()])[0]
            return label, conf.cpu().item()

    def process_file(self, input_path):
        """文件预测+人工标注"""
        try:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
            text, truncated = sanitize_text(raw_text)
            pred_label, confidence = self.predict(text)

            # 输出结果文件
            filename = os.path.basename(input_path)
            output_path = os.path.join("./sort_output", f"{os.path.splitext(filename)[0]}_result.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("文本分类结果\n=============\n")
                f.write(f"文件名称: {filename}\n预测类别: {pred_label}\n置信度: {confidence:.2%}\n")
                if truncated:
                    f.write("(提示：输入已截断至500字)\n")
                f.write("\n文本预览:\n" + text)

            # 人工标注
            if os.environ.get('NO_LABEL_PROMPT') != '1' and ENABLE_HUMAN_LABELING:
                classes = list(getattr(self.label_encoder, 'classes_', []))
                if classes:
                    print("可选类别: " + ", ".join(classes))
                gt = input("请输入实际类别（可多个，英文逗号分隔，回车跳过）: ").strip()
                valid, new, invalid = _parse_human_labels(gt, classes, ALLOW_NEW_LABELS, STRICT_LABEL_CHECK)
                if invalid:
                    print(f"⚠️ 忽略未知标签: {', '.join(invalid)}")
                if new:
                    print(f"ℹ️ 已记录新增标签: {', '.join(new)}")

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                row = dict(
                    source='file', filename=filename, pred_label=pred_label,
                    confidence=f"{confidence:.4f}",
                    human_labels="|".join(valid), new_labels="|".join(new),
                    invalid_labels="|".join(invalid), timestamp=ts
                )
                _append_csv(row)
                _append_jsonl({**row, "text": text})

            print(f"📄 已处理 {filename} → {os.path.basename(output_path)}")
            return output_path
        except Exception as e:
            print(f"❌ 文件 {os.path.basename(input_path)} 出错: {e}")


def process_terminal_input(classifier):
    """终端输入模式"""
    print("\n📝 请输入要分类的文本（空行结束）")
    print("=" * 60)
    lines = []
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
    if not lines:
        print("❕ 未输入任何文本")
        return

    text, truncated = sanitize_text("\n".join(lines))
    label, conf = classifier.predict(text)
    print("\n" + "=" * 60)
    print(f"预测类别: {label}\n置信度: {conf:.2%}")
    if truncated:
        print("(提示：输入已截断至500字)")
    print("=" * 60)

    # 人工标注同样适用
    if ENABLE_HUMAN_LABELING and os.environ.get('NO_LABEL_PROMPT') != '1':
        classes = list(getattr(classifier.label_encoder, 'classes_', []))
        if classes:
            print("可选类别: " + ", ".join(classes))
        gt = input("请输入实际类别（可多个，英文逗号分隔，回车跳过）: ").strip()
        valid, new, invalid = _parse_human_labels(gt, classes, ALLOW_NEW_LABELS, STRICT_LABEL_CHECK)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        row = dict(
            source='terminal', filename='-', pred_label=label,
            confidence=f"{conf:.4f}",
            human_labels="|".join(valid), new_labels="|".join(new),
            invalid_labels="|".join(invalid), timestamp=ts
        )
        _append_csv(row)
        _append_jsonl({**row, "text": text})
        print("✅ 标注已保存")

def print_welcome_message():
    print("=" * 60)
    print("欢迎使用 BERT 文本分类工具")
    print("=" * 60)
    print("1 - 文件输入（./sort_input）")
    print("2 - 终端输入")
    print("3 - 切换是否允许新增类别")
    print("4 - 切换严格模式")
    print("0 - 退出")
    print("=" * 60)

def main():
    create_necessary_dirs()
    print_welcome_message()
    classifier = TextClassifier(best_model_path)

    while True:
        choice = input("请输入选项 (0-4): ").strip()
        if choice == '1':
            for f in os.listdir('./sort_input'):
                if f.endswith('.txt'):
                    classifier.process_file(os.path.join('./sort_input', f))
        elif choice == '2':
            process_terminal_input(classifier)
        elif choice == '3':
            globals()['ALLOW_NEW_LABELS'] = not ALLOW_NEW_LABELS
            print(f"已切换: 允许新增类别 = {'是' if ALLOW_NEW_LABELS else '否'}")
        elif choice == '4':
            globals()['STRICT_LABEL_CHECK'] = not STRICT_LABEL_CHECK
            print(f"已切换: 严格模式 = {'是' if STRICT_LABEL_CHECK else '否'}")
        elif choice == '0':
            print("👋 已退出程序。")
            break
        else:
            print("无效选项，请重新输入。")

if __name__ == "__main__":
    main()
