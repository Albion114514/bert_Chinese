# -*- coding: utf-8 -*-
"""
读取 train_bert.py 导出的日志文件，对关键变量进行“讲解式”打印：
- 每步的学习率/损失/梯度范数：帮助理解优化过程（warmup→下降；loss 降；梯度稳定）
- 每次评估的指标：accuracy / macro_f1
- [CLS] 向量：展示其形状与范数（说明“用 [CLS] 做句子级表示”的合理性）
- 注意力权重：展示层数/头数/序列长度，并打印一个小样例的注意力“top-k”对齐
"""
import os
import json
import numpy as np
import pandas as pd
from argparse import ArgumentParser


def main():
    ap = ArgumentParser()
    ap.add_argument("--run_dir", type=str, default="./runs/bert_sst2", help="train_bert.py 的 output_dir")
    ap.add_argument("--topk", type=int, default=3, help="打印注意力时，展示每个 token 注意最多的 top-k 目标")
    args = ap.parse_args()

    log_dir = os.path.join(args.run_dir, "logs")
    step_csv = os.path.join(log_dir, "train_steps.csv")
    eval_csv = os.path.join(log_dir, "eval_logs.csv")
    inter_dir = os.path.join(log_dir, "interpret")

    print("=== 1) 训练过程（每步）变量 ===")
    if os.path.exists(step_csv):
        df = pd.read_csv(step_csv)
        print(df.head(10))
        print("\n摘要：")
        print(" - steps 总数:", len(df))
        print(" - 学习率范围: [{:.2e}, {:.2e}]".format(df["learning_rate"].min(), df["learning_rate"].max()))
        print(" - 训练 loss（最近10步均值）:", df["loss"].tail(10).mean())
        print(" - 梯度全局范数（中位数）:", df["grad_global_norm"].median())
        print("说明：学习率通常先 warmup 后衰减（参照 Transformer 的 warmup 策略），"
              "loss 在训练初期快速下降，梯度范数保持在可控范围表示训练稳定。")
    else:
        print("未找到:", step_csv)

    print("\n=== 2) 评估指标（每次 evaluate） ===")
    if os.path.exists(eval_csv):
        df = pd.read_csv(eval_csv)
        for _, row in df.iterrows():
            metrics = json.loads(row["metrics_json"])
            print(f"step={row['global_step']}: acc={metrics.get('eval_accuracy')}, f1={metrics.get('eval_macro_f1')}")
        print("说明：分类任务通常以 accuracy 为主，也可结合 macro_f1 评估类别不均衡情况。")
    else:
        print("未找到:", eval_csv)

    print("\n=== 3) 可解释性产物（CLS 向量 & 注意力权重） ===")
    tokens_path = os.path.join(inter_dir, "tokens.json")
    cls_path = os.path.join(inter_dir, "cls_embeddings.npy")
    attn_path = os.path.join(inter_dir, "attentions.npy")

    if all(os.path.exists(p) for p in [tokens_path, cls_path, attn_path]):
        with open(tokens_path, "r", encoding="utf-8") as f:
            tk = json.load(f)
        texts = tk["texts"]
        tokens = tk["tokens"]  # list[batch] of token list
        cls = np.load(cls_path, allow_pickle=True)
        attentions = np.load(attn_path, allow_pickle=True)  # object array: list[num_layers] of arrays

        print(f"- 示例文本（batch={len(texts)}）:")
        for i, t in enumerate(texts):
            print(f"  [{i}] {t}")

        print(f"- [CLS] 向量形状: {cls.shape}（batch, hidden_size），示例第0条L2范数: {np.linalg.norm(cls[0]):.4f}")
        print("  说明：BERT 微调分类时使用 [CLS] 的表征做分类头输入（论文图1）。")

        # 打印第0条样本，第1层第1个注意力头的 top-k 关注
        # attentions: list[num_layers], each -> (batch, heads, seq, seq)
        num_layers = len(attentions)
        l0 = attentions[0]  # 第1层
        batch, heads, seq, _ = l0.shape
        head_0 = l0[0, 0]  # (seq, seq)
        print(f"- 注意力形状示例：层数={num_layers}，首层形状={l0.shape}=(batch, heads, seq, seq)")

        seq_tokens = tokens[0]
        print("  第0条样本，第1层第1头：为每个 token 打印 top-{} 注意目标".format(args.topk))
        for i in range(min(seq, 12)):  # 只展示前12个token，避免太长
            scores = head_0[i]  # 对第 i 个 token 的注意力分布
            top_idx = scores.argsort()[-args.topk:][::-1]
            top_str = ", ".join(f"{seq_tokens[j]}({scores[j]:.3f})" for j in top_idx)
            print(f"    token[{i}] {seq_tokens[i]} -> {top_str}")

        print("说明：多头注意力允许模型在不同子空间关注不同依赖（Transformer 论文第3.2节），"
              "BERT 编码器堆叠多层后，在分类任务中通过 [CLS] 汇聚全局语义。")
    else:
        print("未找到可解释性文件（可能是训练未跑到首次 evaluate）。路径：", inter_dir)


if __name__ == "__main__":
    main()
