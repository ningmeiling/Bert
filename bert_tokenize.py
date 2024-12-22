import torch
import torch.nn as nn
from config import BertConfig
import unicodedata

'''
这段代码实现了一个BertTokenizer类，该类用于对输入文本进行分词、编码并生成BERT模型所需的输入格式
完成分词
'''

class BasicTokenizer:
    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case #do_lower_case 参数用于决定是否将文本中的字符转换为小写
 
    def _run_split_on_punc(self, text):
        """按标点符号分词"""
        output = []
        chars = list(text) # 将输入的字符串 text 转换为字符列表 chars，这样可以逐个字符地进行处理
        i = 0
        start_new_word = True #如果 start_new_word 为 True，说明接下来的字符应该是一个新的词的开始
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(ord(char)):
                output.append([char]) #elf._is_punctuation(ord(char)):: 调用 _is_punctuation 方法判断当前字符是否为标点符号。ord(char) 将字符转换为其对应的 Unicode 编码点
                start_new_word = True #标点符号后，标记下一个字符为新词的开始
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False #表示当前已经开始处理一个词，接下来的字符都属于该词的一部分。
                output[-1].append(char) # 将当前字符 char 添加到 output 中最后一个列表（即当前词）
            i += 1
        return ["".join(x) for x in output] #最后，使用列表推导式将 output 中的每个子列表（每个词的字符列表）连接成一个字符串，返回包含这些字符串的列表。每个元素都是一个分词后的词
 
    def _is_control(self, char):
        """
        判断是否为控制字符
        控制字符包括:
        - C0和C1控制字符 (\x00-\x1F和\x7F-\x9F)
        - 不换行的零宽字符 (\u200B等)
        """
        if char in [0x200B, 0x200C, 0x200D]:  # Zero-width characters
            return True
        if char == 0x0A or char == 0x0D:  # 保留换行符
            return False
        return unicodedata.category(chr(char)).startswith('C')
 
    def _is_whitespace(self, char):
        """
        判断是否为空白字符
        包括:
        - 空格
        - \t \n \r
        - 零宽空格等
        """
        # 零宽空格
        if char == 0x200B:
            return True
        return chr(char).isspace()
 
    def _is_punctuation(self, char):
        """判断是否为标点符号"""
        cp = chr(char)
        # 检查Unicode分类是否为标点符号
        if ((cp >= "!") and (cp <= "/")) or ((cp >= ":") and (cp <= "@")) or \
           ((cp >= "[") and (cp <= "`")) or ((cp >= "{") and (cp <= "~")):
            return True
        cat = unicodedata.category(cp)
        if cat.startswith("P"):
            return True
        return False
    
    def tokenize(self, text):
        """基础分词"""
        text = self._clean_text(text)
        # 中文字符单独成词
        text = self._tokenize_chinese_chars(text)
        # 按空白符分词
        orig_tokens = text.strip().split()
        # 分词结果
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
            split_tokens.extend(self._run_split_on_punc(token))
        return split_tokens
 
    def _clean_text(self, text):
        """清理无效字符"""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self._is_control(cp):
                continue
            if self._is_whitespace(cp):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
 
    def _is_chinese_char(self, cp):
   
    # 中文字符的 Unicode 范围
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  # CJK统一表意文字
            (cp >= 0x3400 and cp <= 0x4DBF) or  # CJK统一表意文字扩展A
            (cp >= 0x20000 and cp <= 0x2A6DF) or  # CJK统一表意文字扩展B
            (cp >= 0x2A700 and cp <= 0x2B73F) or  # CJK统一表意文字扩展C
            (cp >= 0x2B740 and cp <= 0x2B81F) or  # CJK统一表意文字扩展D
            (cp >= 0x2B820 and cp <= 0x2CEAF) or  # CJK统一表意文字扩展E
            (cp >= 0xF900 and cp <= 0xFAFF) or  # CJK兼容表意文字
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  # CJK兼容表意文字补充
            return True
        return False
 
    def _tokenize_chinese_chars(self, text):
        """处理中文字符"""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.extend([" ", char, " "])
            else:
                output.append(char)
        return "".join(output)
 
class WordpieceTokenizer:
    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
 
    def tokenize(self, text):
        """WordPiece分词"""
        output_tokens = []
        for token in text.strip().split():
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
 
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
 
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

vocab = {
    "[CLS]": 0,
    "[SEP]": 1,
    "[PAD]": 2,
    "[MASK]": 3,
    "the": 4,
    "a": 5,
    "of": 6,
    "to": 7,
    "[UNK]": 8,
    # 其他单词
}

# 保存词表到文件
with open("vocab.txt", "w", encoding="utf-8") as f:
    for token, index in vocab.items():
        f.write(f"{token}\n")


class BertTokenizer:
    def __init__(self, vocab_file, config, max_length=None):
        self.vocab = self.load_vocab(vocab_file)  # 加载词表 调用 load_vocab 方法加载词表，并将其存储在 self.vocab 中。self.vocab 是一个字典，映射词汇到其对应的 ID。
        self.max_len = config.max_position_embeddings if max_length is None else max_length #设置最大序列长度
        self.basic_tokenizer = BasicTokenizer() #实例化一个 BasicTokenizer 对象，负责基本的文本预处理，如小写化、去除标点等
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)#实例化一个 WordpieceTokenizer 对象；用加载的词表进行词片（wordpiece）标记化。这种方法可以处理未登录词
        # 特殊token的ID
        self.cls_token_id = self.vocab['[CLS]']
        self.sep_token_id = self.vocab['[SEP]']
        self.pad_token_id = self.vocab['[PAD]']
        self.mask_token_id = self.vocab['[MASK]']
 
    def load_vocab(self, vocab_file): #从词表文件中加载词汇表
        """加载词表"""
        vocab = {}
        with open(vocab_file, 'r', encoding='utf-8') as f: #创建一个空的字典vocab，用于存储词汇表中的词汇及其对应的索引
            for i, line in enumerate(f):#使用enumerate函数遍历文件对象f中的每一行，i表示当前行的索引，line表示当前行的内容
                token = line.strip()#使用strip方法去除当前行line的首尾空白字符，并将结果存储在变量token中。strip方法确保只获取词汇，而不包括任何多余的空白字符
                vocab[token] = i #将去除空白字符后的token作为键，将行索引i作为值，存储在字典vocab中。这一步将词汇和它在文件中的行号建立了映射关系
        return vocab
 
    def convert_tokens_to_ids(self, tokens):
        """将token转换为id"""
        #遇到未知token，返回[UNK]的id
        return [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens] #self.vocab.get(token, self.vocab['[UNK]'])：尝试从词汇表self.vocab中获取当前token的ID。如果token不在词汇表中，则返回词汇表中表示未知token [UNK] 的ID
 
    def tokenize(self, text):
        """分词方法"""
        # 1. 基础分词（分句+清理）
        split_tokens = self.basic_tokenizer.tokenize(text)
        
        # 2. WordPiece分词
        output_tokens = []
        for token in split_tokens:
            sub_tokens = self.wordpiece_tokenizer.tokenize(token)
            output_tokens.extend(sub_tokens)
            
        return output_tokens
    
    def encode_plus(self, texts_a, texts_b=None):
        """
        批量处理输入序列
        texts_a: List[str], 第一个句子列表
        texts_b: List[str], 可选，第二个句子列表
        """
        if texts_b is not None:
            assert len(texts_b) == len(texts_a), "两个句子列表长度不匹配"#检查texts_b是否为None。如果texts_b不为None，则断言texts_b和texts_a的长度相等，否则抛出断言错误，提示“两个句子列表长度不匹配”
 
        # 存储批处理结果
        all_input_ids = []
        all_token_type_ids = []
        all_attention_mask = []
 
        # 处理每个样本
        for i in range(len(texts_a)):
            if texts_b is not None:
                encoded = self.encode(texts_a[i], texts_b[i])
            else:
                encoded = self.encode(texts_a[i])
            
            all_input_ids.append(encoded['input_ids'])
            all_token_type_ids.append(encoded['token_type_ids'])
            all_attention_mask.append(encoded['attention_mask'])
 
        # 堆叠为批处理张量
        return {
            'input_ids': torch.stack(all_input_ids),
            'token_type_ids': torch.stack(all_token_type_ids),
            'attention_mask': torch.stack(all_attention_mask)
        }#将批处理结果列表转换为张量（tensor），并返回一个包含这些张量的字典
    
    def encode(self, text_a, text_b=None):
        """
        构造BERT输入序列
        text_a: 第一个句子
        text_b: 第二个句子(可选)
        """
        # 分词并转换为ID
        tokens_a = self.tokenize(text_a)
        
        # 构造输入序列
        tokens = ['[CLS]'] + tokens_a
        segment_ids = [0] * len(tokens)  # 第一个句子的segment id为0 #segment_ids = [0] * len(tokens) 创建一个与tokens长度相同的列表segment_ids，每个元素的值为0，用于表示第一个句子的Segment ID
        
        # 如果有第二个句子
        if text_b is not None:
            tokens_b = self.tokenize(text_b)
            tokens += ['[SEP]'] + tokens_b + ['[SEP]'] #更新segment_ids，第一个[SEP]的Segment ID为0，tokens_b的Segment ID为1，第二个[SEP]的Segment ID也为1
        else:
            tokens += ['[SEP]']
            segment_ids += [0] #segment_ids += [0] 更新segment_ids，[SEP]的Segment ID为0
            
        # 转换为ID
        input_ids = self.convert_tokens_to_ids(tokens) 
        
        # 创建attention mask
        attention_mask = [1] * len(input_ids) ##attention_mask = [1] * len(input_ids) 创建一个与input_ids长度相同的列表attention_mask，每个元素的值为1，用于表示要注意的tokens位置
        
        # 填充或截断到指定长度
        padding_length = self.max_len - len(input_ids)
        if padding_length > 0 :
            input_ids += [self.pad_token_id] * padding_length #将input_ids填充到指定长度，填充值为[PAD]的ID
            attention_mask += [0] * padding_length # 将attention_mask填充到指定长度，填充值为0（表示这些位置不需要注意）
            segment_ids += [0] * padding_length #将segment_ids填充到指定长度，填充值为0。
        else:
            input_ids = input_ids[:self.max_len]
            attention_mask = attention_mask[:self.max_len]
            segment_ids = segment_ids[:self.max_len]
            
        return {
            'input_ids': torch.tensor(input_ids),
            'token_type_ids': torch.tensor(segment_ids),
            'attention_mask': torch.tensor(attention_mask)
        }
config = BertConfig()
# 初始化BertTokenizer
tokenizer = BertTokenizer(vocab_file='vocab.txt', config=config)

# 单个句子分词
text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
print(tokens)

# 将token转换为ID
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)

# 编码单个句子
encoded = tokenizer.encode(text)
print(encoded)

# 批量编码多个句子
texts_a = ["Hello, how are you?", "I'm fine, thank you!"]
encoded_batch = tokenizer.encode_plus(texts_a)
print(encoded_batch)