import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
# 定义GCN模型类
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(1234)
        # 第一层图卷积层，输入特征数为num_features，输出特征数为4
        self.conv1 = GCNConv(num_features, 4)
        # 第二层图卷积层，输入特征数为4，输出特征数为4
        self.conv2 = GCNConv(4, 4)
        # 第三层图卷积层，输入特征数为4，输出特征数为2
        self.conv3 = GCNConv(4, 2)
        # 全连接分类层，输入特征数为2，输出特征数为类别数num_classes
        self.classifier = nn.Linear(2, num_classes)

    def forward(self, x, edge_index):
        # 第一层图卷积操作，传入节点特征x和边索引edge_index
        h = self.conv1(x, edge_index)
        # 使用tanh激活函数
        h = torch.tanh(h)
        # 第二层图卷积操作
        h = self.conv2(h, edge_index)
        # 使用tanh激活函数
        h = torch.tanh(h)
        # 第三层图卷积操作
        h = self.conv3(h, edge_index)
        # 使用tanh激活函数
        h = torch.tanh(h)
        # 分类层
        out = self.classifier(h)
        return out, h
    

    

