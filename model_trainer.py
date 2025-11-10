# model_trainer.py
import os
import torch
import torch.optim as optim  # 优化器
import joblib  # 保存label_encoder用的


class BERTTrainer:
    """
    BERT模型的训练器类，封装了训练、评估、保存模型的功能
    这样写代码看起来清楚点，不然全堆在一起乱
    """
    def __init__(self, model, device, save_path):
        """
        初始化训练器
        - model: 要训练的模型
        - device: 训练用的设备（CPU/GPU）
        - save_path: 模型保存的根目录
        """
        self.model = model  # 模型
        self.device = device  # 设备
        self.save_path = save_path  # 保存路径
        # 创建保存目录，不存在就新建
        os.makedirs(save_path, exist_ok=True)

    def train_epoch(self, train_loader, optimizer, epoch, total_epochs):
        """
        训练一个epoch（把所有训练数据跑一遍）

        参数：
        - train_loader: 训练集加载器
        - optimizer: 优化器（比如AdamW）
        - epoch: 当前是第几个epoch
        - total_epochs: 总共有多少个epoch

        返回：
        - 这个epoch的平均损失
        """
        # 把模型设为训练模式！一定要写，不然训练不了（比如dropout层会生效）
        self.model.train()
        total_loss = 0.0  # 累计损失

        # 循环每个batch
        for batch_idx, batch in enumerate(train_loader):
            # 把数据移到设备上（CPU/GPU）
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 前向传播：把数据喂给模型，得到输出
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels  # 传labels会自动计算loss
            )
            loss = outputs.loss  # 取出损失
            total_loss += loss.item()  # 累加损失（转成Python数字）

            # 反向传播：更新参数
            optimizer.zero_grad()  # 先清空梯度，不然会累加
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数

            # 每10个batch打印一次进度，不然不知道训练到哪了
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)  # 平均损失
                print(
                    f'Epoch [{epoch}/{total_epochs}] | Batch [{batch_idx + 1}/{len(train_loader)}] | Loss: {avg_loss:.4f}')

        # 返回这个epoch的平均损失
        return total_loss / len(train_loader)

    def evaluate(self, val_loader):
        """
        在验证集上评估模型表现

        参数：
        - val_loader: 验证集加载器

        返回：
        - 平均验证损失
        - 验证准确率
        """
        # 把模型设为评估模式！关闭dropout等，结果更稳定
        self.model.eval()
        val_loss = 0.0  # 累计验证损失
        correct_predictions = 0  # 正确的预测数
        total_predictions = 0  # 总预测数

        # 评估时不计算梯度，省内存，速度也快
        with torch.no_grad():
            for batch in val_loader:
                # 数据移到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # 前向传播，得到输出
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss  # 验证损失
                logits = outputs.logits  # 预测的logits（还没转成概率）

                val_loss += loss.item()  # 累加验证损失

                # 计算准确率：找logits里最大的索引作为预测结果
                predictions = torch.argmax(logits, dim=1)
                # 统计正确的数量（预测和标签一样的）
                correct_predictions += torch.sum(predictions == labels).item()
                total_predictions += len(labels)  # 总数量

        # 计算平均验证损失和准确率
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions

        return avg_val_loss, accuracy

    def save_checkpoint(self, model, label_encoder, accuracy, epoch, is_best=False):
        """
        保存模型检查点（包括模型权重、标签编码器、训练信息）

        参数：
        - model: 要保存的模型
        - label_encoder: 标签编码器（预测时要用）
        - accuracy: 当前的准确率
        - epoch: 当前epoch
        - is_best: 是否是目前最好的模型

        返回：
        - 保存的路径
        """
        # 确定保存目录：最好的模型放best_model，其他按epoch号命名
        if is_best:
            model_dir = os.path.join(self.save_path, "best_model")
        else:
            model_dir = os.path.join(self.save_path, f"epoch_{epoch}")

        # 创建目录
        os.makedirs(model_dir, exist_ok=True)

        # 保存模型权重和配置
        model.save_pretrained(model_dir)
        # 保存标签编码器（用joblib）
        joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))

        # 保存一些训练信息，方便以后查看
        info_file = os.path.join(model_dir, "training_info.txt")
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(f"Epoch: {epoch}\n")  # 第几个epoch
            f.write(f"Accuracy: {accuracy:.4f}\n")  # 准确率
            f.write(f"Classes: {list(label_encoder.classes_)}\n")  # 有哪些类别
            f.write(f"Best Model: {is_best}\n")  # 是否是最佳模型

        print(f"模型已保存到: {model_dir}")
        return model_dir  # 返回保存路径


def train_bert_model(
        model,
        train_loader,
        val_loader,
        label_encoder,
        device,
        save_path,
        learning_rate=3e-5,
        epochs=3
):
    """
    训练BERT模型的主函数，把上面的Trainer串起来用

    参数：
    - model: BERT模型
    - train_loader: 训练集加载器
    - val_loader: 验证集加载器
    - label_encoder: 标签编码器
    - device: 设备
    - save_path: 保存路径
    - learning_rate: 学习率（一般3e-5比较合适，别改太大）
    - epochs: 训练轮数（太少欠拟合，太多过拟合，先试试3轮）

    返回：
    - 最佳模型的保存路径
    """
    # 初始化训练器
    trainer = BERTTrainer(model, device, save_path)
    # 优化器用AdamW，BERT一般都用这个
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    best_accuracy = 0.0  # 记录最好的准确率，一开始是0
    best_model_path = None  # 最好的模型路径

    print("开始训练BERT模型...")

    # 循环每个epoch
    for epoch in range(1, epochs + 1):
        print(f"\n{'=' * 60}")
        print(f'Epoch {epoch}/{epochs}')  # 显示当前是第几个epoch
        print(f"{'=' * 60}")

        # 训练阶段：返回这个epoch的平均损失
        train_loss = trainer.train_epoch(train_loader, optimizer, epoch, epochs)
        print(f'训练损失: {train_loss:.4f}')  # 打印训练损失

        # 验证阶段：返回平均损失和准确率
        val_loss, accuracy = trainer.evaluate(val_loader)
        print(f'验证损失: {val_loss:.4f} | 验证准确率: {accuracy:.4f}')  # 打印验证结果

        # 保存当前epoch的模型（不是最好的也保存，万一后面要用）
        trainer.save_checkpoint(model, label_encoder, accuracy, epoch)

        # 如果当前准确率比之前最好的高，就更新最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # 保存为最佳模型
            best_model_path = trainer.save_checkpoint(
                model, label_encoder, accuracy, epoch, is_best=True
            )
            print(f"🎉 新的最佳模型！准确率: {accuracy:.4f}")  # 庆祝一下

    print(f"\n训练完成！")
    print(f"最佳准确率: {best_accuracy:.4f}")
    print(f"最佳模型路径: {best_model_path}")  # 最后打印最佳模型路径

    return best_model_path  # 返回最佳模型路径，供预测用