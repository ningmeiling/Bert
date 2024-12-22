import torch.nn as nn
import torch
import math
from config import BertConfig
'''
位置编码是什么？怎么用？
Bert使用位置编码来提供输入词汇的位置信息，因为transformer架构本身不处理序列顺序

计算方法：
偶数维度：PE（pos，2i）= sin（pos/（1000**（2*i/d）
奇数维度：PE（pos，2i）= cos（pos/（1000**（2*i/d）
其中d是嵌入维度

使用方法：
将每个词的嵌入与其位置编码相加，形成最终输入
'''
##位置编码
class PositionalEncoding(nn.Module): #定义一个PositionalEncoding继承自nn.Module
    def __init__(self, config): #定义了PositionalEncoding类一个初始化方法，接受一个名为config的参数
        super(PositionalEncoding, self).__init__() #调用父类的初始化方法
        max_len = config.max_position_embeddings #从config参数中或许最大长度位置编码
        d_model= config.hidden_size #从传入的 config 参数中获取隐藏层的维度大小
        pe = torch.zeros(max_len, d_model) #创建一个形状为 (100, 512) 的全零张量，用于存储位置编码
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #创建一个包含位置信息的张量，从 0 到 max_len[100,1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) #计算位置编码的除数项1000**（2*i/d
        
        pe[:, 0::2] = torch.sin(position * div_term)#计算偶数位置的正弦值并存储在 pe 中 0::2是切片操作，为0的元素开始，每隔2个元素取一个值
        pe[:, 1::2] = torch.cos(position * div_term)#算奇数位置的正弦值并存储在 pe 中
        
        pe = pe.unsqueeze(0).transpose(0, 1) #调整 pe 的维度，使其形状变为 [100,1,512]#便于提出位置编码所需个数，同时便于广播机制
        self.register_buffer('pe', pe) #使用 register_buffer 方法将位置编码张量 pe 注册为模型的缓冲区
    
    def forward(self, x): #定义了 PositionalEncoding 类的前向传播方法，接受输入张量 x
        
        x = x + self.pe[:x.size(0), :] #提取出batchsize个数的位置编码，在embedding的每一个位置即（512）上以添加位置编码信息;维度设置为1便于广播机制
        return x


def main():
    config = BertConfig()#实数化config
    # 实例化 PositionalEncoding 类
    pe = PositionalEncoding(config)#实例化PositionalEncoding类（即实数化init下面的未知参数）
    input_tensor = torch.rand(2, 10, 768)
    # 将输入传递给 PositionalEncoding 类的实例
    output = pe(input_tensor)#使用已经实例化的 PositionalEncoding 类，即调用PositionalEncoding 的forward
    print(output)

if __name__ == "__main__":
    main()