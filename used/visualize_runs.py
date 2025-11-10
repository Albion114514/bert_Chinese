# -*- coding: utf-8 -*-
"""
visualize_runs.py
读取 train_bert.py 与 explain_runs.py 导出的文件，生成常见配图：
1) 训练过程曲线：loss / learning rate / grad norm
2) 评估指标曲线：accuracy / macro_f1
3) 注意力热力图：任选 layer/head/sample
4) [CLS] 向量降维散点：PCA 或 TSNE

使用示例：
python visualize_runs.py --run_dir ./runs/bert_sst2 --layer 0 --head 0 --sample 0 --use_tsne
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def plot_curve(x, y, xlabel, ylabel, title, out_path):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_step_curves(step_csv, out_dir):
    if not os.path.exists(step_csv):
        print("未找到", step_csv)
        return
    df = pd.read_csv(step_csv)

    # 1) 训练损失
    plot_curve(
        x=df["global_step"], y=df["loss"],
        xlabel="global_step", ylabel="train_loss",
        title="Training Loss vs. Step",
        out_path=os.path.join(out_dir, "train_loss.png"),
    )

    # 2) 学习率
    plot_curve(
        x=df["global_step"], y=df["learning_rate"],
        xlabel="global_step", ylabel="learning_rate",
        title="Learning Rate vs. Step",
        out_path=os.path.join(out_dir, "learning_rate.png"),
    )

    # 3) 梯度全局范数
    if "grad_global_norm" in df.columns:
        plot_curve(
            x=df["global_step"], y=df["grad_global_norm"],
            xlabel="global_step", ylabel="grad_global_norm",
            title="Grad Global Norm vs. Step",
            out_path=os.path.join(out_dir, "grad_norm.png"),
        )


def plot_eval_curves(eval_csv, out_dir):
    if not os.path.exists(eval_csv):
        print("未找到", eval_csv)
        return
    df = pd.read_csv(eval_csv)
    steps = df["global_step"].tolist()
    acc, f1 = [], []
    for _, row in df.iterrows():
        m = json.loads(row["metrics_json"])
        # Hugging Face 的键通常是 eval_accuracy / eval_macro_f1
        acc.append(m.get("eval_accuracy", None))
        f1.append(m.get("eval_macro_f1", None))

    if any(v is not None for v in acc):
        plot_curve(
            x=steps, y=acc,
            xlabel="global_step", ylabel="eval_accuracy",
            title="Eval Accuracy vs. Step",
            out_path=os.path.join(out_dir, "eval_accuracy.png"),
        )
    if any(v is not None for v in f1):
        plot_curve(
            x=steps, y=f1,
            xlabel="global_step", ylabel="eval_macro_f1",
            title="Eval Macro-F1 vs. Step",
            out_path=os.path.join(out_dir, "eval_macro_f1.png"),
        )


def plot_attention_heatmap(attn_path, tokens_path, out_dir, layer=0, head=0, sample=0, max_tokens=32):
    if not (os.path.exists(attn_path) and os.path.exists(tokens_path)):
        print("未找到注意力或tokens文件：", attn_path, tokens_path)
        return
    with open(tokens_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    tokens = meta["tokens"]  # list[batch] of token list
    texts = meta["texts"]

    # attentions.npy 是 object 数组，里头是 list[num_layers] of np.array
    attentions = np.load(attn_path, allow_pickle=True)
    # 取出该层张量：shape = (batch, heads, seq, seq)
    layer_attn = attentions[layer]
    mat = layer_attn[sample, head]  # (seq, seq)

    # 截断可视化长度，避免过长
    seq_len = min(mat.shape[0], max_tokens)
    mat = mat[:seq_len, :seq_len]
    toks = tokens[sample][:seq_len]

    plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.xticks(range(seq_len), toks, rotation=90)
    plt.yticks(range(seq_len), toks)
    plt.title(f"Attention Heatmap (layer={layer}, head={head}, sample={sample})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"attn_heatmap_l{layer}_h{head}_s{sample}.png"))
    plt.close()


def plot_cls_embeddings(cls_path, tokens_path, out_dir, use_tsne=False):
    if not (os.path.exists(cls_path) and os.path.exists(tokens_path)):
        print("未找到 CLS 或 tokens 文件：", cls_path, tokens_path)
        return
    with open(tokens_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    texts = meta["texts"]
    cls = np.load(cls_path, allow_pickle=True)  # (batch, hidden)

    # 降维到 2D
    if use_tsne and cls.shape[0] >= 2:
        reducer = TSNE(n_components=2, init="pca", learning_rate="auto")
    else:
        reducer = PCA(n_components=2)
    emb2d = reducer.fit_transform(cls)

    plt.figure()
    plt.scatter(emb2d[:, 0], emb2d[:, 1])
    for i, t in enumerate(texts):
        plt.annotate(str(i), (emb2d[i, 0], emb2d[i, 1]))
    plt.title("[CLS] Embeddings (2D)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cls_embeddings_2d.png"))
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default="./runs/bert_sst2", help="训练输出目录")
    ap.add_argument("--layer", type=int, default=0, help="注意力层编号（从0开始）")
    ap.add_argument("--head", type=int, default=0, help="注意力头编号（从0开始）")
    ap.add_argument("--sample", type=int, default=0, help="样本编号（interpret中保存的batch索引）")
    ap.add_argument("--use_tsne", action="store_true", help="对 CLS 向量使用 t-SNE（默认 PCA）")
    ap.add_argument("--max_tokens", type=int, default=32, help="注意力热图可视化时的最大token数")
    args = ap.parse_args()

    figs = ensure_dir(os.path.join(args.run_dir, "figs"))
    logs = os.path.join(args.run_dir, "logs")
    inter = os.path.join(logs, "interpret")

    # 1) 训练过程曲线
    plot_step_curves(os.path.join(logs, "train_steps.csv"), figs)

    # 2) 评估指标曲线
    plot_eval_curves(os.path.join(logs, "eval_logs.csv"), figs)

    # 3) 注意力热力图
    plot_attention_heatmap(
        attn_path=os.path.join(inter, "attentions.npy"),
        tokens_path=os.path.join(inter, "tokens.json"),
        out_dir=figs,
        layer=args.layer, head=args.head, sample=args.sample, max_tokens=args.max_tokens
    )

    # 4) [CLS] 向量可视化
    plot_cls_embeddings(
        cls_path=os.path.join(inter, "cls_embeddings.npy"),
        tokens_path=os.path.join(inter, "tokens.json"),
        out_dir=figs,
        use_tsne=args.use_tsne
    )

    print("✅ 图像已输出到：", figs)


if __name__ == "__main__":
    main()
