import torch
import torch.nn as nn
import math
from config import BertConfig
from position import PositionalEncoding
'''
为什么要做BertEmbeddings？怎么做

1，Embedding 是一种将离散数据（如单词、字符、类别等）转换为连续向量表示的方法。这种方法在自然语言处理（NLP）和机器学习的其他领域中广泛应用。具体来说，Embedding 层主要用于以下几个方面：

1） 转换离散数据到连续空间
    离散数据（如单词）通常以整数索引表示。Embedding 层通过查找表的方式将这些索引映射到高维连续向量空间中，使得模型能够更好地处理这些数据。

2） 捕捉语义信息
    在自然语言处理中，Embedding 层能够捕捉到单词之间的语义关系。通过训练，具有相似语义的单词会在嵌入空间中靠得更近。
    比如，“猫”和“狗”可能会有相似的嵌入向量，而“猫”和“汽车”的嵌入向量则会相差较大。

3） 降维和特征提取
    相比于使用独热编码（one-hot encoding），Embedding 层能够显著降低维度，并且可以提取更多的特征信息。独热编码通常会导致维度非常高
    而 Embedding 层则能够在较低维度上表示更多的语义信息。

4） 处理类别数据
    在推荐系统和分类问题中，类别数据（如用户ID、商品ID等）也可以通过 Embedding 层进行处理，从而将这些类别数据映射到连续向量空间中，便于后续模型的训练。

总的来说，Embedding提供了一种比独热编码更好的对输入对象编码的方法。

将输入的 token ID 转换为嵌入向量，并加入了位置编码和句子编码（token type embeddings），并最终通过层归一化和 Dropout 层进行处理
'''
class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 词嵌入层：将词转换为向量
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        # 位置编码：为每个位置生成唯一的向量表示
        self.position_embeddings = PositionalEncoding(config)
        # 段落嵌入：区分不同句子
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)#一个用于生成句子类型（token type）的嵌入层。BERT 模型有两种句子：A 句子和 B 句子。通过这个嵌入层，每个 token 会有一个类型嵌入（比如标识 token 属于句子 A 还是 B）。type_vocab_size 是句子类型的数量（通常是 2），hidden_size 是输出的嵌入向量维度。
        # 层归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
 
    def forward(self, input_ids, token_type_ids=None):
        #input_ids：输入的词id：【batch_size,seq_length】
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        #维度扩展并广播，【seq_length】-》【batch_size,seq_length】
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)#将 position_ids 扩展到与 input_ids 相同的形状。
        # 如果没有提供token_type_ids，则默认全为0（单句情况）
        # token_type_ids：【batch_size,seq_length】
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
 
        words_embeddings = self.word_embeddings(input_ids)#word_embeddings 层将 input_ids 转换为嵌入向量，即将每个 token ID 映射到一个 hidden_size 维的向量
        words_position_embeddings = self.position_embeddings(words_embeddings)#将词嵌入传递给 position_embeddings 层，添加位置编码信息。
        token_type_embeddings = self.token_type_embeddings(token_type_ids)#通过 token_type_embeddings 层将 token_type_ids 转换为对应的嵌入向量
 
        embeddings = words_position_embeddings + token_type_embeddings #将词嵌入、位置嵌入和句子类型嵌入相加，得到最终的嵌入向量。位置编码和句子类型编码通常是加法操作，与词嵌入合并
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

def main():
    config = BertConfig()
    Embedding = BertEmbeddings(config)
    input_ids = torch.randint(0, 30522, (2, 10))  # 随机生成的词 ID，假设词汇表大小为 30522
    output = Embedding(input_ids)
    #print(output)
if __name__ == "__main__":
    main()#只能使用功能，不能被调用