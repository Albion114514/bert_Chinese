# -*- coding: utf-8 -*-
"""
BERT 中文文本分类（可直接跑 THUCNews 子集 TSV）
- 读取: C:\nice_try\PY2026\bert\thucnews\train/dev/test.tsv
- 模型: bert-base-chinese（可换成你已有中文 BERT）
- 记录与讲解：训练/验证Loss、Acc、F1、学习率、梯度范数、预测分布、混淆矩阵、各类报告
- 输出到 outputs/ 目录

运行：
    python train_bert_thucnews.py
可选参数见 argparse 注释
"""

import os, json, math, random, time
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# 如无代理，首次会自动下载模型，需网络
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup

# ---------------- 配置区 ----------------
DATA_DIR = r"C:\nice_try\PY2026\bert\thucnews"
TRAIN_TSV = os.path.join(DATA_DIR, "train.tsv")
DEV_TSV = os.path.join(DATA_DIR, "dev.tsv")
TEST_TSV = os.path.join(DATA_DIR, "test.tsv")
L2I_JSON = os.path.join(DATA_DIR, "label2id.json")

OUT_DIR = r"C:\nice_try\PY2026\bert\outputs"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_DIR = os.path.join(OUT_DIR, "logs")
CKPT_DIR = os.path.join(OUT_DIR, "ckpt")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

SEED = 2026
MODEL_NAME = r"C:\nice_try\PY2026\bert\outputs"  # 可改成你本地模型路径
MAX_LEN = 256  # 可参考 data_profile.json 的 P95
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRAD_ACCUM_STEPS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED);
torch.manual_seed(SEED);
torch.cuda.manual_seed_all(SEED)


# --------------- 工具函数 ---------------
def read_tsv(path: str) -> Tuple[List[str], List[str]]:
    labels, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line: continue
            y, x = line.split("\t", 1)
            labels.append(y.strip());
            texts.append(x.strip())
    return labels, texts


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_csv(path, row_dict: Dict):
    # 简易 CSV 追加器：自动写表头
    import csv
    write_header = (not os.path.exists(path))
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if write_header: w.writeheader()
        w.writerow(row_dict)


# --------------- 数据集 ---------------
class TsvDataset(Dataset):
    def __init__(self, tsv_path: str, tokenizer: BertTokenizerFast, label2id: Dict[str, int], max_len: int):
        labels, texts = read_tsv(tsv_path)
        self.labels = [label2id[y] for y in labels]
        self.texts = texts
        self.tok = tokenizer
        self.maxlen = max_len

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        y = self.labels[idx]
        enc = self.tok(
            text,
            max_length=self.maxlen,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(y, dtype=torch.long)
        return item


# --------------- 训练/评估 ---------------
@torch.no_grad()
def evaluate(model, loader, device) -> Dict:
    model.eval()
    all_preds, all_trues = [], []
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        logits = out.logits
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
        trues = batch["labels"].detach().cpu().tolist()
        all_preds += preds;
        all_trues += trues
    avg_loss = total_loss / max(1, len(loader))
    acc = accuracy_score(all_trues, all_preds)
    macro_f1 = f1_score(all_trues, all_preds, average="macro")
    return {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "preds": all_preds,
        "trues": all_trues
    }


def train():
    # 1) 读标签映射/数据
    with open(L2I_JSON, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)

    # 2) tokenizer / datasets
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME, do_lower_case=False)
    ds_train = TsvDataset(TRAIN_TSV, tokenizer, label2id, MAX_LEN)
    ds_dev = TsvDataset(DEV_TSV, tokenizer, label2id, MAX_LEN)
    ds_test = TsvDataset(TEST_TSV, tokenizer, label2id, MAX_LEN)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dl_dev = DataLoader(ds_dev, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3) 模型
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    ).to(DEVICE)

    # 4) 优化器/调度器
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=LR)
    total_steps = math.ceil(len(dl_train) / GRAD_ACCUM_STEPS) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # 5) 保存超参
    save_json(os.path.join(LOG_DIR, "hparams.json"), {
        "model_name": MODEL_NAME, "max_len": MAX_LEN, "batch_size": BATCH_SIZE,
        "lr": LR, "epochs": EPOCHS, "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO, "grad_accum_steps": GRAD_ACCUM_STEPS,
        "num_labels": num_labels, "seed": SEED
    })

    best_dev = -1.0
    global_step = 0

    # 6) 训练循环（带详细日志）
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(dl_train, 1):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / GRAD_ACCUM_STEPS
            loss.backward()

            # 记录梯度/权重范数（简要）
            if step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            running_loss += loss.item()

            # 每 N step 记录一次
            if global_step % 20 == 0:
                row = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": running_loss / max(1, 20),
                    "lr": scheduler.get_last_lr()[0]
                }
                append_csv(os.path.join(LOG_DIR, "metrics.csv"), row)
                running_loss = 0.0

        # 每个 epoch 结束做一次验证
        dev_metrics = evaluate(model, dl_dev, DEVICE)
        row = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch,
            "step": global_step,
            "dev_loss": dev_metrics["loss"],
            "dev_acc": dev_metrics["acc"],
            "dev_macro_f1": dev_metrics["macro_f1"]
        }
        append_csv(os.path.join(LOG_DIR, "metrics.csv"), row)

        # 保存最优
        score = dev_metrics["macro_f1"]
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "last.pt"))
        if score > best_dev:
            best_dev = score
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best.pt"))

    # 7) 用最佳模型在测试集评估 & 输出更详细报告
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "best.pt"), map_location=DEVICE))
    test_metrics = evaluate(model, dl_test, DEVICE)
    y_true = test_metrics["trues"]
    y_pred = test_metrics["preds"]

    # classification report
    report = classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(num_labels)],
                                   output_dict=True, digits=4)
    save_json(os.path.join(LOG_DIR, "label_report.json"), report)

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_labels)))
    # 保存为 CSV，行列为真实/预测
    import csv
    with open(os.path.join(LOG_DIR, "confusion_matrix.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = [""] + [id2label[i] for i in range(num_labels)]
        w.writerow(header)
        for i in range(num_labels):
            row = [id2label[i]] + list(cm[i])
            w.writerow(row)

    # 预测分布（看是否偏科）
    pred_counter = Counter(y_pred)
    save_json(os.path.join(LOG_DIR, "pred_distribution.json"), {id2label[k]: int(v) for k, v in pred_counter.items()})

    # 打印简表
    print("== TEST ==")
    print(f"loss={test_metrics['loss']:.4f}, acc={test_metrics['acc']:.4f}, macro_f1={test_metrics['macro_f1']:.4f}")
    print("报告与混淆矩阵已保存到 outputs/logs/")


if __name__ == "__main__":
    train()
