
import torch.nn as nn


class EncodeAndDecode(nn.Module):

    def __init__(self, encoder, decoder):
        super(EncodeAndDecode, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, targets):
        input_var, input_len = input_seq
        output_en, hidden_en = self.encoder.forward(input_var, input_len)
        output_de, hidden_de = self.decoder.forward(hidden_en, targets)  # 疑问：targets是否只需考虑最大len?
        # 而不需要考虑每句的lens
        return output_de, hidden_de


