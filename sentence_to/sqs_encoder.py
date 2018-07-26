import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sentence_to.server_request import read_from_url
from sentence_to.sts_data_setup import DataTransformer

from torch.autograd import Variable

# batch_size = 3
# windows = 3
# max_length = 256  # sentence maximum length


class Encoder(nn.Module):

    def __init__(self, vocab_size, embedded_size, out_size):
        """

        Args:
            vocab_size: 词汇数量
            max_length: embedding size 也就是feature 维度
        """
        # TODO: 我觉得这里需要把vocab_size 转化成所建立的最大文本长度。
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedded_size)  # 目的：每传入一句话，用词将这句话表示出来
        #  目前认为，max_length 就是一个文本中的最大词长。
        self.gru = nn.GRU(embedded_size, out_size)

    def forward(self, torch_variable, input_length):
        """

        Args:
            torch_variable: torch.longtensor 型变量

        Returns:

        """
        embedding = self.embedding(torch_variable)
        padded_sequence = pack_padded_sequence(embedding, input_length)
        out_put_first, hidden = self.gru(padded_sequence, None)
        out_put_final, out_put_lengths = pad_packed_sequence(out_put_first)
        return out_put_final, hidden


if __name__ == '__main__':
    res = read_from_url("bj_sentence.txt")[0:1000]
    embedding_size = 125
    encoder_output_size = 125
    data_transformer = DataTransformer(res, use_cuda=False)
    Encoder_pre = Encoder(vocab_size=data_transformer.vocab_size, embedded_size=embedding_size,
                          out_size=encoder_output_size)