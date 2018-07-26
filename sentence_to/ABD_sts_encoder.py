# model/Encoder.py
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VanillaEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, output_size):
        """Define layers for a vanilla rnn encoder 单层rnn解析网络"""
        super(VanillaEncoder, self).__init__()  # 调用父类nn.Module的初始化函数。

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, output_size)  # 这步还没理解。

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)  # 设置好embedding层次后，传入参数
        # input_seqs 应该是tensor 中的Variable 类型
        packed = pack_padded_sequence(embedded, input_lengths)
        packed_outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = pad_packed_sequence(packed_outputs)
        return outputs, hidden

    def forward_a_sentence(self, inputs, hidden=None):
        """Deprecated, forward 'one' sentence at a time which is bad for gpu utilization"""
        embedded = self.embedding(inputs)
        outputs, hidden = self.gru(embedded, hidden)
        return outputs, hidden