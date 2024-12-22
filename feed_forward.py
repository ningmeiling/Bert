import torch
import torch.nn as nn
from config import BertConfig

'''
FeedForward有什么用
FeedForward的功能是对输入的隐藏状态进行线性变换、激活、再变换，并最终应用dropout来增强模型的鲁棒性
'''
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
 
    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

def main():  
    config = BertConfig()
     #实例化 FeedForward 类a
    feedforward = FeedForward(config)
    input_tensor = torch.rand(2, 10, 768)
     # 将输入传递给 FeedForward 层
    output = feedforward(input_tensor)
     # 输出张量的形状
    print(output.shape) 

if __name__ == "__main__":
    main()#只能使用功能，不能被调用






