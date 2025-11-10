# -*- coding: utf-8 -*-
"""
THUCNews（子集）数据分析与格式化脚本
- 读取 C:\nice_try\PY2026\bert\thucnews\sub.txt
- 检查每行格式：label<TAB>text
- 统计标签分布、文本长度分布，清洗空行
- 按标签分层划分 train/dev/test（8/1/1）
- 导出 TSV 与标签映射、数据概览，供后续 BERT 脚本直接使用

运行：
    python format_thucnews.py
"""

import os, json, random, math, re
from collections import Counter, defaultdict

# ---- 修改你的数据根路径 ----
DATA_ROOT = r"C:\nice_try\PY2026\bert\thucnews"
RAW_FILE  = os.path.join(DATA_ROOT, "sub.txt")

OUT_DIR   = DATA_ROOT  # 直接输出到 thucnews 目录
TRAIN_TSV = os.path.join(OUT_DIR, "train.tsv")
DEV_TSV   = os.path.join(OUT_DIR, "dev.tsv")
TEST_TSV  = os.path.join(OUT_DIR, "test.tsv")
L2I_JSON  = os.path.join(OUT_DIR, "label2id.json")
PROFILE   = os.path.join(OUT_DIR, "data_profile.json")

SEED = 2026
random.seed(SEED)

def clean_text(s: str) -> str:
    """
    极简清洗：去掉多余空白、控制符。保留中文/英文/数字/标点。
    你也可以按需加：全角->半角、繁简转换、去除URL等。
    """
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def read_raw():
    """
    逐行读取 sub.txt
    期望：每行 'label<TAB>text'
    容错：如果没有 TAB 或文本过短，丢弃。
    返回 list[(label, text)]
    """
    data = []
    bad_lines = 0
    with open(RAW_FILE, "r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                bad_lines += 1
                continue
            if "\t" not in line:
                # 有些行可能是脏数据，直接跳过
                bad_lines += 1
                continue
            label, text = line.split("\t", 1)
            label = label.strip()
            text  = clean_text(text)
            if not label or len(text) < 5:
                bad_lines += 1
                continue
            data.append((label, text))
    return data, bad_lines

def stratified_split(items, labels, ratio=(0.8, 0.1, 0.1)):
    """
    分层划分：同一标签内按比例切分，保证类分布一致
    返回：train/dev/test 列表（元素为索引）
    """
    by_label = defaultdict(list)
    for idx, y in enumerate(labels):
        by_label[y].append(idx)
    for y in by_label:
        random.shuffle(by_label[y])

    idx_train, idx_dev, idx_test = [], [], []
    for y, idxs in by_label.items():
        n = len(idxs)
        n_train = int(n * ratio[0])
        n_dev   = int(n * ratio[1])
        # 剩余给 test
        cur = 0
        idx_train += idxs[cur:cur+n_train]; cur += n_train
        idx_dev   += idxs[cur:cur+n_dev];   cur += n_dev
        idx_test  += idxs[cur:]
    random.shuffle(idx_train)
    random.shuffle(idx_dev)
    random.shuffle(idx_test)
    return idx_train, idx_dev, idx_test

def write_tsv(path, items, labels, indices):
    with open(path, "w", encoding="utf-8") as f:
        for i in indices:
            f.write(f"{items[i][0]}\t{items[i][1]}\n")

def main():
    print(f"[INFO] 读取：{RAW_FILE}")
    items, bad = read_raw()
    assert len(items) > 0, "没有读到有效样本，请检查 sub.txt 的编码/格式。"
    print(f"[INFO] 有效样本：{len(items)}；丢弃行数：{bad}")

    labels = [y for y, _ in items]
    label_counter = Counter(labels)
    print("[INFO] 标签分布：", label_counter)

    # 标签 -> id
    uniq_labels = sorted(label_counter.keys())
    label2id = {lab: i for i, lab in enumerate(uniq_labels)}

    # 文本长度统计（以字符数粗略估）
    lengths = [len(x[1]) for x in items]
    avg_len = sum(lengths)/len(lengths)
    p95_len = sorted(lengths)[int(len(lengths)*0.95)-1]
    print(f"[INFO] 文本平均长度：{avg_len:.1f}，P95：{p95_len}")

    # 分层划分
    idx_train, idx_dev, idx_test = stratified_split(items, labels, ratio=(0.8, 0.1, 0.1))
    print(f"[INFO] 划分 -> train:{len(idx_train)}, dev:{len(idx_dev)}, test:{len(idx_test)}")

    # 写出 TSV
    write_tsv(TRAIN_TSV, items, labels, idx_train)
    write_tsv(DEV_TSV,   items, labels, idx_dev)
    write_tsv(TEST_TSV,  items, labels, idx_test)

    # 保存映射与数据概览
    with open(L2I_JSON, "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    profile = {
        "total_samples": len(items),
        "dropped_lines": bad,
        "num_labels": len(uniq_labels),
        "labels": label_counter,
        "avg_len": avg_len,
        "p95_len": p95_len,
        "split": {"train": len(idx_train), "dev": len(idx_dev), "test": len(idx_test)},
        "seed": SEED
    }
    with open(PROFILE, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    print(f"[OK] 已输出：\n  {TRAIN_TSV}\n  {DEV_TSV}\n  {TEST_TSV}\n  {L2I_JSON}\n  {PROFILE}")

if __name__ == "__main__":
    main()
