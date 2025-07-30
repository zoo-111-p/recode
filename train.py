
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score

from model.kggmu import GatedClassifier as KGGMu

# GatedBimodal, MLPGenreClassifier

from model.kglabel import GatedClassifier
# 特征适配器
class FeatureAdapter(nn.Module):
    def __init__(self, input_dim, output_dim=512):
        super(FeatureAdapter, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        return x

# 调整特征维度
def adjust_feature_dimensions(vis_features, text_features, kg_features):
    # 将视觉特征展平
    vis_features_flat = vis_features.view(vis_features.size(0), -1)

    # 定义特征适配器
    vis_adapter = FeatureAdapter(3 * 224 * 224, 512)
    text_adapter = FeatureAdapter(77, 512)
    kg_adapter = FeatureAdapter(23, 512)

    # 调整特征维度
    adjusted_vis_features = vis_adapter(vis_features_flat)
    adjusted_text_features = text_adapter(text_features)
    adjusted_kg_features = kg_adapter(kg_features)

    return adjusted_vis_features, adjusted_text_features, adjusted_kg_features

# class clipweightClassifier(nn.Module):
#     def __init__(self, vis_dim, text_dim, kg_dim, n_classes, hidden_size):
#         super(clipweightClassifier, self).__init__()
#         # 使用GatedClassifier进行分类
#         self.gated_classifier = KGGMu.GatedClassifier(vis_dim, text_dim, kg_dim, n_classes, hidden_size)
#
#     def forward(self, vis_input, text_input, kg_input):
#         # 直接调用GatedClassifier进行前向传播
#         y_hat, gate_output = self.gated_classifier(vis_input, text_input, kg_input)
#         return y_hat, gate_output

#
# 分类
class clipweightClassifier(nn.Module):
    def __init__(self, vis_dim, text_dim, kg_dim, n_classes, hidden_size):
        super(clipweightClassifier, self).__init__()

        # 特征适配器
        self.vis_adapter = FeatureAdapter(vis_dim, 512)
        self.text_adapter = FeatureAdapter(text_dim, 512)
        self.kg_adapter = FeatureAdapter(kg_dim, 512)

        # 全连接层
        self.classifier = nn.Linear(hidden_size * 3, n_classes)
        self.relu = nn.ReLU()

    def forward(self, vis_features, text_features, kg_features):
        batch_size = vis_features.size(0)

        # 调整视觉特征的维度
        vis_features = vis_features.view(batch_size, -1)
        vis_output = self.vis_adapter(vis_features)

        # 调整文本特征的维度
        text_output = self.text_adapter(text_features)

        kg_output = self.kg_adapter(kg_features)

        # 将所有特征连接在一起
        combined_features = torch.cat((vis_output, text_output, kg_output), dim=1)
        combined_features = combined_features.view(batch_size, -1)  # 展平特征
        logits = self.classifier(combined_features)
        return logits


# 训练和评估模型
class clipweightTrainer:
    def __init__(self, train_data, dev_data, test_data, model, clipmodel, args, logger, writer, target_names):
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model.to(args.device)
        self.clipmodel = clipmodel.to(args.device)
        self.args = args
        self.logger = logger
        self.writer = writer
        self.target_names = target_names
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        best_dev_f1 = 0.0  # Track the best Macro F1 score
        for epoch in range(self.args.num_epochs):
            self.model.train()
            total_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            # 遍历批次
            for batch in tqdm(self.train_data, desc=f"Training Epoch {epoch + 1}"):
                vis_features, text_features, kg_features, labels = (
                    batch[0].to(self.args.device),
                    batch[1].to(self.args.device),
                    batch[2].to(self.args.device),
                    batch[3].to(self.args.device)
                )

                # 确保特征是 Float 类型
                vis_features = vis_features.float()
                text_features = text_features.float()
                kg_features = kg_features.float()

                # print(f"Current Batch Labels: {labels.tolist()}")

                # 调整特征维度
                vis_output, text_output, kg_output = adjust_feature_dimensions(vis_features, text_features, kg_features)

                # 将 labels 转换为一维
                if labels.dim() > 1 and labels.size(1) > 1:
                    labels = torch.argmax(labels, dim=1)
                assert labels.dim() == 1, f"labels should be 1D, but got {labels.dim()}D"

                # print(f"Current Batch Labels: {labels.tolist()}")

                # 清零梯度
                self.optimizer.zero_grad()

                # 前向传播
                outputs = self.model(vis_output, text_output, kg_output)
                # outputs = self.model(vis_features, text_features, kg_features)

                # 计算准确率
                _, preds = torch.max(outputs, 1)
                correct_predictions += torch.sum(preds == labels).item()
                total_predictions += labels.size(0)

            # 记录训练损失和准确率
            train_accuracy = correct_predictions / total_predictions
            avg_loss = total_loss / len(self.train_data)
            print(f"Epoch {epoch + 1} - Training Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}")
            if self.writer:
                self.writer.add_scalar("Training Loss", avg_loss, epoch)
                self.writer.add_scalar("Training Accuracy", train_accuracy, epoch)

            # 在开发集上进行评估
            dev_micro_f1, dev_macro_f1 = self.evaluate(self.dev_data)
            print(f"Epoch {epoch + 1} - Dev Micro F1: {dev_micro_f1 :.4f}, Macro F1: {dev_macro_f1 :.4f}")
            if self.writer:
                self.writer.add_scalar("Dev Micro F1", dev_micro_f1, epoch)
                self.writer.add_scalar("Dev Macro F1", dev_macro_f1, epoch)

            # 根据开发集 Macro F1 保存最佳模型
            if dev_macro_f1 > best_dev_f1:
                best_dev_f1 = dev_macro_f1
                if self.args.save_path:
                    torch.save(self.model.state_dict(), os.path.join(self.args.save_path, 'best_model.pth'))

    def evaluate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                vis_features, text_features, kg_features = (
                    batch[0].to(self.args.device),
                    batch[1].to(self.args.device),
                    batch[2].to(self.args.device)
                )
                labels = batch[3].to(self.args.device)

                # 确保特征是 Float 类型
                vis_features = vis_features.float()
                text_features = text_features.float()
                kg_features = kg_features.float()

                # 调整特征维度
                vis_output, text_output, kg_output = adjust_feature_dimensions(vis_features, text_features, kg_features)

                # 将 labels 转换为一维
                if labels.dim() > 1 and labels.size(1) > 1:
                    labels = torch.argmax(labels, dim=1)
                assert labels.dim() == 1, f"labels should be 1D, but got {labels.dim()}D"

                # 前向传播
                outputs = self.model(vis_output, text_output, kg_output)
                _, preds = torch.max(outputs, 1)

                # outputs = self.model(vis_features, text_features, kg_features)
                # _, preds = torch.max(outputs, 1)

                # 收集所有预测和标签
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算 Micro F1 和 Macro F1
        micro_f1 = f1_score(all_labels, all_preds, average='micro')
        macro_f1 = f1_score(all_labels, all_preds, average='macro')

        print(f"Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}")

        return micro_f1, macro_f1

    def test(self):
        print("Testing the best model...")
        self.model.load_state_dict(torch.load(os.path.join(self.args.save_path, 'best_model.pth')))
        test_micro_f1, test_macro_f1 = self.evaluate(self.test_data)
        print(f"Test Micro F1: {test_micro_f1:.4f}, Macro F1: {test_macro_f1:.4f}")
        if self.writer:
            self.writer.add_scalar("Test Micro F1", test_micro_f1)
            self.writer.add_scalar("Test Macro F1", test_macro_f1)
        return test_micro_f1, test_macro_f1


































