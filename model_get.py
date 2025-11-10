# model_get.py
# 从transformers导入自动加载模型和分词器的工具
from transformers import AutoModel, AutoTokenizer

# 要下载的模型名称，这里用的是bert-base-chinese（中文BERT基础版）
# 可以换成其他模型，比如"hfl/chinese-roberta-wwm-ext"
model_name = "bert-base-chinese"
# 本地保存路径，自己改！比如r"D:\models\bert-base-chinese"
# 路径别太长，不然可能报错
local_dir = r"C:\nice_try\PY2026\bert\local_bert"

# 下载并加载模型和分词器，同时会自动保存到local_dir
# 第一次运行会下载，可能有点慢，耐心等
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 手动再保存一次，确保万无一失
# 保存后，local_dir里会有config.json、pytorch_model.bin、vocab.txt等文件
model.save_pretrained(local_dir)
tokenizer.save_pretrained(local_dir)

# 提示：下载完后，训练时的MODEL_PATH就填这个local_dir