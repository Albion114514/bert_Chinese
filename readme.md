以下是该项目的 **README.md**，包含完整说明、使用方法与标注扩展功能：

---

# 🧠 BERT 中文文本分类工具

基于 HuggingFace `transformers` 的中文 **BERT 文本分类系统**，支持本地训练、文件批量预测、终端输入预测，以及 **人工标注扩展** 功能（单/多标签、自定义标签、严格模式、重训练准备）。

---

## 🚀 功能特性

### ✅ 基础功能

* 使用 BERT 模型进行中文文本分类。
* 训练与预测全流程自动化。
* 配置文件统一管理参数（路径、批大小、学习率等）。

### 🧩 输入模式

* **文件输入模式**：将 `.txt` 文件放入 `./sort_input` 文件夹，系统会批量读取并分类。
* **终端输入模式**：直接在命令行输入文本，立即返回分类结果。

### 🧾 输出

* 分类结果自动保存至 `./sort_output/` 文件夹：

  * 单条结果保存为 `*_result.txt`
  * 所有标注汇总保存为：

    * `annotations.csv`（表格格式）
    * `annotations.jsonl`（结构化格式，便于重训练）

### 🧍‍♀️ 人工标注扩展（New!）

* 支持预测后输入**实际类别**（单标签或多标签，英文逗号分隔）。
* 可显示模型训练时的类别列表。
* 未知类别可选择：

  * **严格模式**：拒绝未知标签；
  * **扩展模式**：允许新增标签，仅记录到日志中。
* 新增菜单项：

  ```
  3 - 切换是否允许新增类别 (ALLOW_NEW_LABELS)
  4 - 切换严格校验模式 (STRICT_LABEL_CHECK)
  ```
* 环境变量控制：

  * `NO_LABEL_PROMPT=1` → 关闭人工标注交互（批处理模式）
  * 默认开启 `ENABLE_HUMAN_LABELING = True`

---

## 🧰 项目结构

```
.
├── config.json             # 全局配置文件
├── data_loader.py          # 数据加载与预处理模块
├── model_get.py            # 模型下载与保存脚本
├── model_trainer.py        # 模型训练逻辑
├── train.py                # 训练主程序
├── predict.py              # 预测主程序（含人工标注）
├── predict_sample.py       # 示例预测脚本
├── requirements.txt        # 依赖库清单
├── sort_input/             # 输入文件夹（需自行创建）
├── sort_output/            # 输出与标注文件夹
│   ├── *_result.txt
│   ├── annotations.csv
│   └── annotations.jsonl
```

---

## ⚙️ 安装与环境

```bash
pip install -r requirements.txt
```

依赖核心库：

* `torch >= 2.2`
* `transformers >= 4.42`
* `scikit-learn >= 1.2`
* `pandas >= 2.0`

---

## 🧑‍💻 训练模型

1. 准备数据集（THUCNews 格式）：

   ```
   标签\t文本内容
   体育\t科比27分湖人止连败...
   财经\t央行宣布下调存款准备金率...
   ```

2. 修改 `config.json`：

   ```json
   {
     "MODEL_PATH": "./local_bert",
     "DATA_PATH": "./thucnews/cnews.train.txt",
     "SAVE_MODEL_PATH": "./output",
     "BEST_MODEL_PATH": "./output/best_model",
     "MAX_LENGTH": 128,
     "BATCH_SIZE": 8,
     "EPOCHS": 3,
     "LEARNING_RATE": 3e-5,
     "TEST_SIZE": 0.2,
     "NUM_CLASSES": 10
   }
   ```

3. 执行训练：

   ```bash
   python train.py
   ```

---

## 🔍 预测与标注

```bash
python predict.py
```

菜单示例：

```
1 - 处理 ./sort_input 文件夹中的文件
2 - 终端输入文本
3 - 切换是否允许新增类别 (当前：是)
4 - 切换严格校验模式 (当前：否)
0 - 退出程序
```

---

## 🧾 输出示例

`sort_output/sample_result.txt`

```
文本分类结果
=============
文件名称: sample.txt
预测类别: 体育
置信度: 95.34%
(提示：输入已按要求截断至500字)
实际类别(有效): 体育
```

`sort_output/annotations.csv`

| source | filename   | pred\_label | confidence | human\_labels | new\_labels | invalid\_labels | timestamp        |
| :----- | :--------- | :---------- | :--------- | :------------ | :---------- | :-------------- | :--------------- |
| file   | sample.txt | 体育          | 0.9534     | 体育            |             |                 | 20251109\_143212 |

---

## ⚙️ 可选环境变量

| 变量                      | 说明               | 默认值   |
| :---------------------- | :--------------- | :---- |
| `NO_LABEL_PROMPT`       | 禁用人工标注交互（设为 `1`） | 否     |
| `ENABLE_HUMAN_LABELING` | 是否启用人工标注功能       | True  |
| `ALLOW_NEW_LABELS`      | 是否允许新增标签（扩展模式）   | True  |
| `STRICT_LABEL_CHECK`    | 是否启用严格标签校验       | False |

---

## 🧩 后续扩展建议

* 增加模型微调功能（可基于标注结果重新训练）。
* 将 `annotations.jsonl` 转换为 HuggingFace `datasets` 格式重用。
* 加入 Web UI（如 Streamlit）可视化预测结果。
