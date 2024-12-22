import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertConfig, BertModel, BertTokenizer

def test_bert():
    # 1. 配置参数
    config = BertConfig(
        vocab_size=21128,  # 中文BERT词表大小
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-12
    )

    # 2. 创建模型
    print('bert model:')
    model = BertModel(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(model)
    model.to(device)

    # 3. 准备测试数据
    texts_a = [
        "我爱你，你爱我吗",
        "今天天气真好",
        "机器学习很有趣"
    ]
    texts_b = [
        "是的，我也爱你",
        "确实是个好天气",
        "深度学习更有趣"
    ]

    # 4. 数据预处理
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')#加载预训练的中文BERT分词器（bert-base-chinese）。这个分词器会根据中文BERT的词汇表（vocabulary）将输入文本转化为token id，提供对文本的分词功能
    input_ids = []
    token_type_ids = []
    attention_mask = []
    '''
tokenizer.encode_plus： tokenizer.encode_plus 会对每对句子进行编码，返回一个包含多个字段的字典。关键字段包括：

input_ids: 一个表示输入文本的张量，每个词被转换成词汇表中的索引。
token_type_ids: 一个标识句子A和句子B的张量，句子A对应0，句子B对应1。
attention_mask: 一个标识哪些token是有效的，哪些是填充的张量。有效token的mask值为1，填充的mask值为0。
    '''
    for text_a, text_b in zip(texts_a, texts_b):#将 texts_a 和 texts_b 中的句子成对组合成一个迭代器。每次循环中，text_a 和 text_b 会依次取出相应的句子对（例如："我爱你" 和 "你爱我吗"）
        encoded = tokenizer.encode_plus(
            text_a,
            text_b,
            add_special_tokens=True,
            padding='max_length',  # padding到最大长度
            truncation=True,       # 截断
            max_length=128,        # 设置最大长度
            return_tensors='pt'    # 返回pytorch tensor格式 指定返回结果为 PyTorch tensor 格式，这样可以直接在PyTorch模型中使用
        )
        # 将编码结果添加到列表
        input_ids.append(encoded['input_ids'])#返回的字典中包含 input_ids，它是一个张量，表示文本的token id（每个token都被映射到一个整数，表示词汇表中的位置）
        token_type_ids.append(encoded['token_type_ids'])#：表示句子A和句子B的标识。通常用于区分不同的句子对（0表示句子A，1表示句子B）
        attention_mask.append(encoded['attention_mask'])#这是一个张量，标识哪些token是有效的（值为1）和哪些是padding（值为0）。它帮助模型忽略padding部分

    # 将列表转换为张量 (使用 torch.cat 拼接为单个 Tensor)
    input_ids = torch.cat(input_ids, dim=0)#将所有句子对的 input_ids 连接起来，形成一个大的张量
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)

    # 5. 创建数据集和数据加载器
    dataset = TensorDataset(input_ids, token_type_ids, attention_mask) #TensorDataset 将数据封装成一个可以传入模型的数据集
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True) #创建一个数据加载器，用于按批次加载数据，batch_size=2 表示每批次加载2个样本，shuffle=True 表示打乱数据顺序

    # 6. 测试模型
    model.eval()#将模型设置为评估模式，禁用dropout等训练时的特性
    with torch.no_grad():#在此上下文中，不计算梯度，以节省内存和加速推理；这对于推理（inference）阶段非常有用，因为在推理过程中，我们并不需要对模型的参数进行更新。因此，禁用梯度计算可以减少内存消耗和加速推理过程
        for batch in dataloader:#dataloader 是一个数据加载器，它在每次迭代时会返回一个批次的数据。在推理过程中，我们按批次处理数据，这样可以更高效地利用内存和计算资源。每个 batch 包含一组输入数据，通常包括 input_ids, token_type_ids 和 attention_mask。
            input_ids, token_type_ids, attention_mask = [t.to(device) for t in batch]#将批次数据从 CPU 移动到 GPU（如果可用），即将数据转移到模型所在的设备上（例如 cuda 或 cpu）。
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )
            
            # 模型输出是一个包含多个元素的元组，按顺序为 (last_hidden_state, pooler_output)
            sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
            
            # 打印输出形状
            print(f"Sequence output shape: {sequence_output.shape}")  # [batch_size, seq_len, hidden_size] 是所有token的输出，形状为 [batch_size, seq_len, hidden_size]
            print(f"Pooled output shape: {pooled_output.shape}")     # [batch_size, hidden_size] 是针对 [CLS] 标记的输出，通常用于分类任务，形状为 [batch_size, hidden_size]
            
            # 打印第一个样本的[CLS]标记的表示
            print(f"First sample [CLS] representation:\n{pooled_output[0][:10]}\n")

# 执行测试函数
test_bert()


