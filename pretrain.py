import torch
import torch.nn as nn
from config import BertConfig
from bert_model import BertModel
'''
BertForPreTraining 类封装了整个BERT预训练任务，包括掩码语言模型（MLM）和下一句预测（NSP）任务。
BertPredictionHeadTransform、BertLMPredictionHead 和 BertPreTrainingHeads 这些类负责定义BERT预训练任务所需的头部模块

交叉熵损失：计算分类概率
'''
#处理隐藏层输出；该类将BERT的隐藏层输出进行全连接变换、激活和归一化处理，以便后续任务（如掩码语言模型）能正确处理这些输出 
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)#定义一个全连接层（dense）。这个层的输入和输出维度都为 config.hidden_size，即BERT的隐藏层维度。它将BERT的隐藏状态进行线性变换
        self.transform_act_fn = nn.GELU()#定义一个激活函数 transform_act_fn，这里使用的是 GELU
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)#定义一个层归一化（LayerNorm）层。层归一化用于稳定训练，并有助于模型的收敛性。它会对输入数据进行归一化，使其均值为0，标准差为1。LayerNorm 在BERT中被用来提高性能
 
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        #breakpoint()
        hidden_states = self.transform_act_fn(hidden_states)#对经过全连接层变换后的 hidden_states 应用 GELU 激活函数。这个步骤引入了非线性，使得模型可以学习更加复杂的映射关系
        #breakpoint()
        hidden_states = self.LayerNorm(hidden_states)#应用层归一化（LayerNorm），对激活后的 hidden_states 进行归一化。这有助于防止梯度爆炸/消失问题，并加速模型的训练
        breakpoint()
        return hidden_states


#用于掩码语言模型：【batch_size,seq_length,hidden_size】->【batch_size,seq_length,vocab_size】该类基于BERT的隐藏层输出，计算每个位置的预测词汇（即掩码语言模型的输出），输出形状为 [batch_size, seq_length, vocab_size]。
class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
 
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        #breakpoint()
        hidden_states = self.decoder(hidden_states) + self.bias
        breakpoint()
        return hidden_states

#seq_relationship_score用于下一句预测任务：【batch_size,hidden_size】->【batch_size,2】
#prediction_scores用于掩码语言模型：【batch_size,seq_length,hidden_size】->【batch_size,seq_length,vocab_size】
class BertPreTrainingHeads(nn.Module):#用于BERT的预训练任务头，计算掩码语言模型（MLM）和下一句预测（NSP）任务的输出
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)#这个对象负责处理BERT模型的输出，以计算 掩码语言模型（Masked Language Model，MLM）的预测分数
        self.seq_relationship = nn.Linear(config.hidden_size, 2)#它是一个线性层，用于计算 **下一句预测（Next Sentence Prediction, NSP）**任务的分数
        #该线性层的输入是BERT模型的 池化输出（pooled_output），其维度是 [batch_size, hidden_size]，输出维度为 2，表示二分类任务（即判断句子B是否是句子A的下一句）
    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        breakpoint()
        return prediction_scores, seq_relationship_score

class BertForPreTraining(nn.Module):#主要的功能是在预训练过程中计算掩码语言模型（Masked Language Model, MLM）和下一句预测（Next Sentence Prediction, NSP）的损失
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, 
                masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
 
        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)## prediction_scores: [batch_size, num_classes], labels: [batch_size]定义一个交叉熵损失函数 loss_fct，忽略索引为 -1 的标签（即不计算某些位置的损失）即被mask掉的值；I am learning [MASK] machine learning
            #掩码损失用于训练模型预测被 [MASK] 替换的词语
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), #masked_lm_labels 是每个词的真实标签，形状为 [batch_size, seq_length]，也是词汇表中的索引。需要调整为 [batch_size * seq_length]，并计算与 prediction_scores 中相应位置的交叉熵损失
                                    masked_lm_labels.view(-1))#将 prediction_scores 形状变为 [batch_size * seq_length, vocab_size]；将 masked_lm_labels 形状变为 [batch_size * seq_length]
            #下一句预测损失用于训练模型判断两个句子是否是连续的
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), 
                                        next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            breakpoint()
            return total_loss
        else:
            return prediction_scores, seq_relationship_score

def main():
    config = BertConfig()
    head = BertPredictionHeadTransform(config)
    mphead = BertLMPredictionHead(config)
    trhead = BertPreTrainingHeads(config)
    pretra = BertForPreTraining(config)
    input_tensor = torch.rand(2, 10, 768)
    pool_tensor = torch.rand(2, 768)
    input_ids = torch.randint(0, 30522, (2, 10))
    head(input_tensor)
    mphead(input_tensor)
    BertPreTrainingHeads(config)(input_tensor,pool_tensor)
    output = pretra(input_ids )
    print(output[1].shape)



if __name__ == "__main__":
    main()#只能使用功能，不能被调用