# -*- coding: utf-8 -*-
"""
训练/微调 BERT 做文本二分类（默认 SST-2），并记录关键变量以便复盘与讲解。
- 核心库：Hugging Face Transformers / Datasets / Accelerate
- 关键记录：训练损失、学习率、梯度范数；评估指标；示例预测；
           以及模型的 [CLS] 向量与注意力权重（用于“原理理解”章节的可解释性展示）。
- 超参默认值与范围参考 BERT 论文微调建议：lr∈{5e-5,3e-5,2e-5}，batch∈{16,32}，epochs∈{2,3,4}。
"""

import os
import math
import csv
import json
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.nn.utils import clip_grad_norm_

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)
from datasets import load_dataset
dataset = load_dataset("seamew/ChnSentiCorp")

# =========================
# 一些小工具：度量与保存函数
# =========================

def seed_everything(seed: int = 42):
    """为了可复现性：固定一切随机种子。"""
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    """
    Hugging Face Trainer 规范的评估函数：
    - 输入：模型在eval上的 logits 与 labels
    - 输出：一个 dict，会被写入“trainer_state.json”和日志
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}


def save_json(obj: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_csv_row(path: str, header: List[str], row: List[Any]):
    """追加一行到 CSV（若文件不存在则写入表头）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


# =========================
# 自定义回调：记录训练过程变量
# =========================
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl


class LoggingCallback(TrainerCallback):
    """
    训练时记录：
    - 每个 step 的 loss、学习率、梯度范数（便于讲解优化过程）
    - 每个 eval 的指标
    - 在首次评估时，抓取一小批数据的注意力权重和 [CLS] 向量，落盘做“可解释性展示”
    """

    def __init__(self, log_dir: str, model, tokenizer, sample_texts: Optional[List[str]] = None):
        self.log_dir = log_dir
        self.model = model
        self.tokenizer = tokenizer
        self.sample_texts = sample_texts or ["This movie is great!", "The plot was boring..."]
        self._saved_interpret_once = False

        # 预创建路径
        os.makedirs(self.log_dir, exist_ok=True)
        self.step_log_csv = os.path.join(self.log_dir, "train_steps.csv")
        self.eval_log_csv = os.path.join(self.log_dir, "eval_logs.csv")

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """
        step 结束时，记录：
        - 当前 global_step
        - 当前学习率（从 scheduler 读）
        - 最近一个 step 的 loss
        - 梯度范数（global norm）
        """
        # 记录学习率：Trainer内部维护了 scheduler
        lr = None
        if kwargs.get("optimizer") is not None and hasattr(kwargs["optimizer"], "param_groups"):
            # 一般只有一个 param_group
            lr = kwargs["optimizer"].param_groups[0]["lr"]

        # loss：state.log_history 里也有，但这里直接取状态中的 recent_loss 更直观
        loss = state.log_history[-1]["loss"] if state.log_history and "loss" in state.log_history[-1] else None

        # 梯度范数：手动计算（需要模型参数上有 grad）
        grad_norm = 0.0
        total = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                grad_norm += param_norm ** 2
                total += 1
        grad_norm = math.sqrt(grad_norm) if total > 0 else None

        append_csv_row(
            self.step_log_csv,
            header=["global_step", "learning_rate", "loss", "grad_global_norm"],
            row=[state.global_step, lr, loss, grad_norm],
        )

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        """
        每次 evaluate 时，追加评估指标。
        并在第一次评估时额外导出一小批样本的注意力权重与 CLS 向量。
        """
        append_csv_row(
            self.eval_log_csv,
            header=["global_step", "metrics_json"],
            row=[state.global_step, json.dumps(metrics)],
        )

        # 只做一次可解释性导出，避免产生过多文件
        if not self._saved_interpret_once:
            self._saved_interpret_once = True
            self._export_interpretability()

    def _export_interpretability(self):
        """
        1）对 sample_texts 做 tokenizer → model 前向，开启 output_attentions & output_hidden_states
        2）保存：
            - attentions：一个 list[h][layer] 的张量（形状 ~ [batch, heads, seq, seq]）
            - cls_embeddings：最后一层隐状态中 [CLS] 的向量（batch, hidden_size）
        """
        self.model.eval()
        # 暂时开启返回注意力与隐状态
        old_attn = getattr(self.model.config, "output_attentions", False)
        old_hidd = getattr(self.model.config, "output_hidden_states", False)
        self.model.config.output_attentions = True
        self.model.config.output_hidden_states = True

        with torch.no_grad():
            batch = self.tokenizer(
                self.sample_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.model.device)
            outputs = self.model(**batch)
            # attentions: list of [batch, num_heads, seq_len, seq_len]
            attentions = outputs.attentions  # 长度= num_layers
            # hidden_states: list of [batch, seq_len, hidden]
            hidden_states = outputs.hidden_states  # 长度= num_layers+1（含embedding层）
            last_hidden = hidden_states[-1]  # 最后一层
            cls_embeddings = last_hidden[:, 0, :].detach().cpu().numpy()  # [CLS] 在位置0

        # 落盘：注意力取第1层/第1头为例，做一个小型热力图矩阵（保存为 .npy 供后续解释脚本读取）
        os.makedirs(os.path.join(self.log_dir, "interpret"), exist_ok=True)
        np.save(os.path.join(self.log_dir, "interpret", "cls_embeddings.npy"), cls_embeddings)
        # 也保存全部注意力（体积稍大，但便于后续分析）
        # 结构：list[num_layers], 每个元素为 (batch, heads, seq, seq)
        # 先转 CPU 再保存
        attn_cpu = [a.detach().cpu().numpy() for a in attentions]
        np.save(os.path.join(self.log_dir, "interpret", "attentions.npy"), np.array(attn_cpu, dtype=object))

        # 保存对应的 token 序列，便于映射注意力到词
        tokens = [self.tokenizer.convert_ids_to_tokens(x) for x in batch["input_ids"].cpu().tolist()]
        save_json({"tokens": tokens, "texts": self.sample_texts}, os.path.join(self.log_dir, "interpret", "tokens.json"))

        # 还原配置
        self.model.config.output_attentions = old_attn
        self.model.config.output_hidden_states = old_hidd


# =========================
# 自定义 Trainer（可选）：在每步后裁剪梯度 & 方便记录
# =========================
class MyTrainer(Trainer):
    """
    继承 Trainer，注入梯度裁剪（clip_grad_norm）与更稳健的 logging 钩子。
    这样新手更容易理解“梯度爆炸/稳定训练”的概念。
    """

    def __init__(self, max_grad_norm: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_grad_norm = max_grad_norm

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        直接复用父类，保持清晰：分类任务 => 交叉熵
        """
        return super().compute_loss(model, inputs, return_outputs)

    def training_step(self, model, inputs):
        """
        单步训练：
        - 前向：得到 loss
        - 反向：loss.backward()
        - 梯度裁剪：clip_grad_norm_
        - 优化器 step & scheduler step
        - 清空梯度
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_contextManager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()

        loss.backward()

        # 梯度裁剪（关键：稳定训练，避免梯度爆炸）
        clip_grad_norm_(model.parameters(), self._max_grad_norm)

        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        self.state.global_step += 1
        return loss.detach()


# =========================
# 数据预处理函数
# =========================
def get_datasets_and_tokenizer(model_name: str, dataset_name: str, text_cols: List[str], label_col: str, max_length: int):
    """
    - dataset_name: e.g., "sst2"（英文），或 "seamew/ChnSentiCorp"（中文）
    - text_cols: 文本列名（单句任务则传一个列名；句对任务可传两个列名）
    - label_col: 标签列名
    """
    # 加载数据集（SST-2：datasets.load_dataset("glue","sst2")；ChnSentiCorp：datasets.load_dataset("seamew/ChnSentiCorp")）
    if dataset_name.lower() in ["sst2", "glue/sst2"]:
        raw = load_dataset("glue", "sst2")
    else:
        raw = load_dataset(dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # 标准化：把原始样本映射到 {'text' 或 'text_pair', 'labels'} 形式
    def _preprocess(batch):
        if len(text_cols) == 1:
            texts = batch[text_cols[0]]
            enc = tokenizer(texts, truncation=True, padding=False, max_length=max_length)
        else:
            enc = tokenizer(batch[text_cols[0]], batch[text_cols[1]], truncation=True, padding=False, max_length=max_length)
        enc["labels"] = batch[label_col]
        return enc

    processed = raw.map(_preprocess, batched=True, remove_columns=raw["train"].column_names)
    return processed, tokenizer


# =========================
# 主函数
# =========================
def main():
    parser = argparse.ArgumentParser()
    # 基本参数
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="可选 'bert-base-uncased'（英文）或 'bert-base-chinese'（中文）")
    parser.add_argument("--dataset", type=str, default="sst2",
                        help="英文：sst2；中文：seamew/ChnSentiCorp 等")
    parser.add_argument("--text_cols", type=str, default="sentence",
                        help="文本列名，单列用逗号分隔传一个，如 'sentence'；句对如 'sentence1,sentence2'")
    parser.add_argument("--label_col", type=str, default="label", help="标签列名")
    parser.add_argument("--output_dir", type=str, default="./runs/bert_sst2",
                        help="输出目录（模型权重、日志、解释性文件都会在此处）")
    # 训练超参（默认值符合 BERT 论文微调推荐范围）
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)          # {16,32} 推荐
    parser.add_argument("--epochs", type=int, default=3)                # {2,3,4} 推荐
    parser.add_argument("--learning_rate", type=float, default=3e-5)    # {5e-5, 3e-5, 2e-5}
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="是否开启混合精度")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["epoch", "steps"])
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["epoch", "steps"])
    parser.add_argument("--evaluation_steps", type=int, default=500, help="当 eval_strategy=steps 时生效")
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--log_layer_grads_every", type=int, default=0,
                        help=">0 时，每隔 N steps 记录一次逐层参数梯度范数到 logs/layer_grads.csv")
    parser.add_argument("--max_layers_log", type=int, default=999,
                        help="最多记录多少个模块/参数（避免超大模型输出过多）")
    args = parser.parse_args()

    # 1) 准备环境
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 2) 准备数据与分词器
    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    dataset, tokenizer = get_datasets_and_tokenizer(
        model_name=args.model_name,
        dataset_name=args.dataset,
        text_cols=text_cols,
        label_col=args.label_col,
        max_length=args.max_length,
    )

    # 3) 加载预训练模型（分类任务头）
    #    这里直接用 BERT 论文提出的“在 [CLS] 上接一个线性层做分类”的微调方式（见图1）:contentReference[oaicite:4]{index=4}
    num_labels = 2  # 默认二分类（SST-2/ChnSentiCorp）
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    # 为了可解释性导出，在评估时临时打开：output_attentions / output_hidden_states
    model.config.output_attentions = False
    model.config.output_hidden_states = False

    # 4) DataCollator：按需动态 padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5) 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        evaluation_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        eval_steps=(args.evaluation_steps if args.eval_strategy == "steps" else None),
        save_steps=(args.evaluation_steps if args.save_strategy == "steps" else None),
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=args.fp16,
        report_to=["none"],  # 关闭wandb等
    )

    # 6) Trainer：使用自定义的 MyTrainer（带梯度裁剪），并挂载 LoggingCallback 记录变量
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        max_grad_norm=args.max_grad_norm,
    )

    # 记录用的样例句子（英文/中文自动切换）
    sample_texts = (
        ["This movie is great!", "The plot was boring..."]
        if "uncased" in args.model_name
        else ["这部电影太棒了！", "剧情有点无聊……"]
    )
    trainer.add_callback(LoggingCallback(log_dir=os.path.join(args.output_dir, "logs"),
                                         model=model, tokenizer=tokenizer, sample_texts=sample_texts))

    # 7) 训练与评估
    trainer.train()
    metrics = trainer.evaluate()
    print("Final Eval:", metrics)
    save_json(metrics, os.path.join(args.output_dir, "final_eval.json"))

    # 8) 保存最佳模型与分词器
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 9) 额外：保存一些示例预测，帮助“讲解模型输出”
    test_split = "test" if "test" in dataset else "validation"
    preds = trainer.predict(dataset[test_split])
    pred_labels = np.argmax(preds.predictions, axis=-1).toli_
