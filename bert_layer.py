import torch
import torch.nn as nn
from config import BertConfig
import math
from mult_head import MultiHeadAttention
from feed_forward import FeedForward
from config import BertConfig
'''
层归一化：假设你有一个形状为 2×3
2×3 的矩阵（2个样本，每个样本3个特征）。层归一化会对每个样本的3个特征进行标准化，确保每个样本内的均值为0，方差为1。
批次归一化：假设你有一个形状为 2×3 的矩阵（2个样本，每个样本3个特征）。批次归一化会对3个特征的所有样本进行标准化，确保每个特征的均值为0，方差为1。

BertLayer有什么用？
BertLayer 是BERT模型中的一个核心模块，主要用于实现BERT的基本功能，即通过多头自注意力和前馈神经网络对输入的序列进行处理。BertLayer 是堆叠起来形成BERT模型的基本单元
'''

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.intermediate = FeedForward(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)#初始化一个层归一化层 attention_norm，它会对注意力模块的输出进行标准化。config.hidden_size 是隐藏层的维度，config.layer_norm_eps 是防止除零错误的一个小常数
 
    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_norm(attention_output + hidden_states)
        layer_output = self.intermediate(attention_output)
        layer_output = self.output_norm(layer_output + attention_output)
        return layer_output

def main():
    config = BertConfig()
    bertlayer = BertLayer(config)
    input_tensor = torch.rand(2, 10, 768)
    output = bertlayer(input_tensor)
    #print(output)
if __name__ == "__main__":
    main()#只能使用功能，不能被调用