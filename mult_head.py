import torch
import torch.nn as nn
import math
from config import BertConfig
'''
多头注意力机制是什么？怎么用？
是Transformer模型中的一个核心概念，旨在通过并行计算多个不同的“注意力头”来增强模型对信息的捕捉能力，进而提升对复杂模式和长距离依赖的理解
注意力机制的核心思想在于，当模型处理输入数据时，不是平等地对待所有数据，而是根据一定的规则选择性地关注某些重要信息。这一机制通过Query、Key、Value三个元素来实现，它们分别代表了查询请求、相关性衡量标准和实际数据内容。
Attention(Q,K,V) = [softmax(QK^T)/d**(1/2)]*V
'''
class MultiHeadAttention(nn.Module):#定义了一个 MultiHeadAttention 类，继承自 torch.nn.Module
    def __init__(self, config):#初始化多头注意力层。接受一个 config 参数
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
 
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)#将hidden states映射到所有注意力头上
 
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size = hidden_states.size(0)
        #形式转换：【batch_size,seq_length,hidden_size】-》【batch_size,seq_length,num_attention_heads,attention_head_size】
        query_layer = self.query(hidden_states).view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1,2)#-1是自动推断维度
        key_layer =self.key(hidden_states).view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1,2)
        value_layer =self.value(hidden_states).view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1,2)#转换维度是为了目的是将 num_attention_heads 维度移到 seq_len 维度之前，以便于在每个注意力头上并行计算注意力分数
 
        # 点乘得到注意力得分
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        #防止元素过大，在softmax中会变成0或者1
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)#[2，8，10，10]表示第 i 个位置（词）的第 j 个位置（词）之间的注意力得分
        
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
 
        # softmax得到注意力概率
        attention_probs = torch.softmax(attention_scores,dim=-1)#将在最后一个维度（即 seq_len 维度）上应用 softmax。这个操作的目的是计算每个词对其他词的“注意力”分布，使得每个词对其他词的注意力权重总和为 1
        attention_probs = self.dropout(attention_probs)#dropout 会在每次前向传播时，以一定的概率将输入的一部分元素置为 0
 
        context = torch.matmul(attention_probs, value_layer).transpose(1,2).contiguous().view(batch_size, -1, self.all_head_size)#将注意力概率与 value_layer 相乘，得到最终的上下文信息，即通过注意力分数计算当前位置值。然后通过 transpose(1, 2) 和 .view() 变换形状；比如在计算某个词时，它不仅仅依赖于自己，还依赖于与其他词的关系
        
        return self.out(context)
def main():
    config = BertConfig()
    multi_head_attention = MultiHeadAttention(config)
    hidden_states = torch.randn(2, 10, 768)
    # 调用 MultiHeadAttention 的 forward 方法，得到输出
    output = multi_head_attention(hidden_states)
    print(output.shape)  # 输出的形状

if __name__ == "__main__":
    main()