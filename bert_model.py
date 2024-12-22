import torch
import torch.nn as nn
from config import BertConfig
from embedding import BertEmbeddings
from bert_layer import BertLayer
'''
这段代码定义了一个 BertModel 类，包含了以下主要组件：词嵌入层、编码器（多个BertLayer）、池化层，并定义了权重初始化方法。
BertModel 是 BERT 模型的实现，主要用于处理文本输入并生成嵌入向量。它通过以下几个步骤：
词嵌入层：将 token ID 转换为向量。
编码器层：通过多层 Transformer 编码器提取 token 之间的关系。
池化层：从输出中提取 [CLS] token 的向量，通常用于分类任务。
'''
class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])#nn.ModuleList 用于存储模块列表；通过 ModuleList 存储多个 BertLayer 层，并将它们按顺序组织在一起
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size) #这一行定义了一个线性层（全连接层），用于池化操作。hidden_size 表示隐藏层的维度，这里定义了一个线性变换，输入和输出维度相同
        self.pooler_activation = nn.Tanh()#这一行定义了一个激活函数，使用 Tanh 对池化输出进行非线性变换
        
        self.apply(self._init_weights) #这一行调用 self.apply() 方法，它会遍历模型中的所有子模块并应用 _init_weights 方法。_init_weights 方法用于初始化模型的权重参数
 
    def _init_weights(self,module):#定义了一个权重初始化方法。根据不同的模块类型（nn.Linear, nn.Embedding, nn.LayerNorm），方法会初始化它们的权重
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()#对于 nn.LayerNorm 类型的模块初始化偏置为 0
            module.weight.data.fill_(1.0)#权重为 1
            
                
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
 
        embedding_output = self.embeddings(input_ids, token_type_ids)
        
        all_encoder_layers = []
        hidden_states = embedding_output
        for layer_module in self.encoder:#这里的循环遍历所有的 BERT 层（BertLayer），将 hidden_states 传递到每一层中，得到每一层的输出。每次计算都会更新 hidden_states
            hidden_states = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)#将每一层的输出添加到 all_encoder_layers 列表中
 
        sequence_output = hidden_states#将最终的 hidden_states 作为 sequence_output（即序列输出），它的形状为 (batch_size, seq_length, hidden_size)
        #输入序列的构造：[CLS] Token1 Token2 Token3... [SEP]
        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))#这一行从 sequence_output 中提取第一个 token 对应的隐藏状态（通常是 [CLS] token），然后通过 pooler 线性层和 Tanh 激活函数得到 pooled_output。最终的 pooled_output 形状为 (batch_size, hidden_size)，通常用于分类任务
        #将 sequence_output[:, 0]（即 [CLS] 标记的表示）传入池化层 self.pooler。池化层的作用是将 [CLS] 标记的表示通过一个线性变换，输出形状为 [batch_size, hidden_size]。
        #pooled_output:【batch_size,hidden_size】
        #sequence_output:【batch_size,seq_length,hidden_size】
        #Tanh 是一种常用的非线性激活函数，将输入值映射到 [-1, 1] 范围内
        #池化也可以用last hidden state的所有token的embedding求均值
        #breakpoint()
        return sequence_output, pooled_output

def main():
    config = BertConfig()
    bertmodel = BertModel(config)
    for name, param in bertmodel.named_parameters():  #来遍历模型中的所有参数并打印它们
        #print(param)
        print(name)
    input_ids = torch.randint(0, 30522, (2, 10))
    output = bertmodel(input_ids)
    print(bertmodel)
    print(type(output))

if __name__ == "__main__":
    main()#只能使用功能，不能被调用