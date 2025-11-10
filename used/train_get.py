import os
import shutil
import pandas as pd

# 目标目录
target_dir = r"C:\nice_try\PY2026\bert\thucnews"


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"已创建目录: {path}")


def process_thucnews(raw_data_path, save_dir):
    """
    处理THUCNews原始数据（需先从官网下载并解压）
    原始数据结构：每个类别一个文件夹，内含多个文本文件
    """
    create_directory(save_dir)
    all_texts = []

    # 遍历所有类别文件夹
    for category in os.listdir(raw_data_path):
        category_path = os.path.join(raw_data_path, category)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().strip()
                    if text:  # 过滤空文本
                        all_texts.append(text)

    # 保存为文本文件（每行一段文本，适合BERT预训练）
    with open(os.path.join(save_dir, "thucnews_corpus.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_texts))
    print(f"THUCNews语料已处理完成，保存至: {save_dir}")


if __name__ == "__main__":
    # 注意：需先从THUCTC官网下载THUCNews数据集并解压，填写解压后的路径
    raw_thucnews_path = r"C:\nice_try\PY2026\bert\thucnews"  # 例如：r"C:\downloads\THUCNews"
    process_thucnews(raw_thucnews_path, target_dir)