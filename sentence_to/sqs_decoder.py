import torch.nn as nn
import torch
import random

from torch.autograd import Variable

# batch_size = 3
# windows = 3
# max_length = 256


class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, max_sentence, use_cuda, teacher_ratio, sos_id):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_sentence
        self.embedding = nn.Embedding(output_size, hidden_size)  # 目的：每传入一句话，用词将这句话表示出来
        #  目前认为，max_length 就是一个文本中的最大词长。
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.sos_id = sos_id
        self.use_cuda = use_cuda
        self.softmax = nn.LogSoftmax(1)
        self.teacher_ratio = teacher_ratio
        self.linear = nn.Linear(hidden_size, output_size)

    def forward_step(self, decode_input, hidden_info):
        batch_size = decode_input.size(1)
        decode_embedding = self.embedding(decode_input)
        decode_embedding.view(1, batch_size, self.hidden_size)  # 这步没理解它的意思 把input 变成(1* barch_size* hidden_size)
        # 如果不修改传入格式的话，会不会出问题，可以后续测一下。
        # 感觉像是为了帮助后面sequence[0] 压缩时间维度信息，从3维到2维做准备的。
        outputs, hidden_layer = self.gru(decode_embedding, hidden_info)
        outputs = outputs.squeeze(0)
        output_result = self.softmax(self.linear(outputs))
        return output_result, hidden_layer

    def forward(self, context_vector, targets):
        batch_size = context_vector.size(1)
        start_flag = Variable(torch.LongTensor([[self.sos_id]*batch_size]))
        decode_hidden = context_vector
        # 传入的这个targets 是（）元组型tuple变量
        targets_var, target_len = targets
        max_target_length = max(target_len)

        decode_output = Variable(torch.zeros(max_target_length, batch_size, self.output_size))

        # if self.use_cuda:
        #     start_flag = start_flag.cuda()
        #     decode_output = decode_output.cuda()

        use_teacher_ratio = True if random.random() > self.teacher_ratio else False
        for t in range(max_target_length):
            decode_on_t, decode_hidden = self.forward_step(start_flag, decode_hidden)
            decode_output[t] = decode_on_t
            if use_teacher_ratio:
                start_flag = targets_var[t].unsqueeze(0)
                print('this is unsqueezeed', start_flag, start_flag.size())
                # 因为target的维度是 time * batch 上面语句获得的是t时刻的预期结果。
            else:
                print('guess run')
                start_flag = self.decoder_index(decode_on_t)
                print('this is decode_flag', start_flag, start_flag.size())

        return decode_output, decode_hidden

    def decoder_index(self, decoder_inputs):
        value, index = torch.Tensor.topk(decoder_inputs, 1)
        index = index.transpose(0, 1)
        # 这里输出的index是 decoder_inputs tensor 向量中最大概率的那一列（那一行代表的元素）的原始index（数值）表示，并不是位置角标。
        if self.use_cuda:
            index = index.cuda()
        return index

    def evaluation(self, context_vector):
        batch_size = context_vector.size(1)
        start_flag = Variable(torch.LongTensor([[self.sos_id] * batch_size]))
        decode_output = Variable(torch.zeros(self.max_length, batch_size, self.output_size))
        decode_hidden = context_vector

        for t in range(self.max_length):
            decode_on_t, decode_hidden = self.forward_step(start_flag, decode_hidden)
            decode_output[t] = decode_on_t
            start_flag = self.decoder_index(decode_on_t)

        return self.decode_to_indices(decode_output)

    def decode_to_indices(self, decode_outputs):
        decode_info = []
        batch_size = decode_outputs.size(1)
        decode_outputs = decode_outputs.transpose(0, 1)

        for i in range(batch_size):
            likelihood_max = self.decoder_index(decode_outputs[i])
            decode_info.append(likelihood_max.data[0])
        return decode_info

    # Todo:查一下这个.data的含义




