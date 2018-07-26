# model/Decoder.py
import random
import torch
import torch.nn as nn
from torch.autograd import Variable


class VanillaDecoder(nn.Module):

    def __init__(self, hidden_size, output_size, max_length, learning_ratio, sos_id, use_cuda):
        """Define layers for a vanilla rnn decoder"""
        super(VanillaDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)  # 线性变化模型结构定义
        self.log_softmax = nn.LogSoftmax(1)  # work with NLLLoss = CrossEntropyLoss
        # 添加 dim = 1 变量

        self.max_length = max_length
        self.teacher_forcing_ratio = learning_ratio
        self.sos_id = sos_id
        self.use_cuda = use_cuda

    def forward_step(self, inputs, hidden):
        # inputs: (time_steps=1, batch_size)
        batch_size = inputs.size(1)  # 列
        embedded = self.embedding(inputs)
        embedded.view(1, batch_size, self.hidden_size)  # S = T(1) x B x N
        rnn_output, hidden = self.gru(embedded, hidden)  # S = T(1) x B x H
        # TODO:猜测：rnn_output 是xt或者说rt在t时刻变换的结果，hidden是添加了前t时刻信息的记忆矩阵
        rnn_output = rnn_output.squeeze(0)  # squeeze the time dimension
        # 将(1,n)维结构压缩成n维，去掉第一位的1维。
        # 与之对应的还有unsqueeze 将n维的torch.tensor型数据转化成为(1,n)维度
        output = self.log_softmax(self.out(rnn_output))
        # S = B x O  # 将输入的hi xi做一个线性变换 然后再套入一个log_softmax
        return output, hidden

    def forward(self, context_vector, targets):

        # Prepare variable for decoder on time_step_0
        target_vars, target_lengths = targets
        print('this is target: ', target_vars, target_vars.size())
        batch_size = context_vector.size(1)
        decoder_input = Variable(torch.LongTensor([[self.sos_id] * batch_size]))
        print('this is start_flag: ', decoder_input, decoder_input.size())

        # Pass the context vector
        decoder_hidden = context_vector

        max_target_length = max(target_lengths)
        decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, self.output_size))
        # (time_steps, batch_size, vocab_size)

        if self.use_cuda:
            decoder_input = decoder_input.cuda()
            decoder_outputs = decoder_outputs.cuda()

        use_teacher_forcing = True if random.random() > self.teacher_forcing_ratio else False

        # Unfold the decoder RNN on the time dimension
        for t in range(max_target_length):
            print('the max target length is: ', t)
            decoder_outputs_on_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            print('this is decode_hidden', decoder_hidden, decoder_hidden.size())
            decoder_outputs[t] = decoder_outputs_on_t
            if use_teacher_forcing:
                decoder_input = target_vars[t].unsqueeze(0)
                print('hhhhhhhhhhhhhhhhhh', decoder_input, decoder_input.size())
            else:
                decoder_input = self._decode_to_index(decoder_outputs_on_t)
            return decoder_outputs, decoder_hidden

    def evaluation(self, context_vector):
        batch_size = context_vector.size(1)  # get the batch size
        decoder_input = Variable(torch.LongTensor([[self.sos_id] * batch_size]))
        decoder_hidden = context_vector

        decoder_outputs = Variable(torch.zeros(
            self.max_length,
            batch_size,
            self.output_size
        ))  # (time_steps, batch_size, vocab_size)

        if self.use_cuda:
            decoder_input = decoder_input.cuda()
            decoder_outputs = decoder_outputs.cuda()

        # Unfold the decoder RNN on the time dimension
        for t in range(self.max_length):
            decoder_outputs_on_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs[t] = decoder_outputs_on_t
            decoder_input = self._decode_to_index(decoder_outputs_on_t)  # select the former output as input

        return self._decode_to_indices(decoder_outputs)

    def _decode_to_index(self, decoder_output):
        """
        evaluate on the logits, get the index of top1
        :param decoder_output: S = B x V or T x V
        """
        value, index = torch.topk(decoder_output, 1)
        index = index.transpose(0, 1)  # S = 1 x B, 1 is the index of top1 class
        if self.use_cuda:
            index = index.cuda()
        return index

    def _decode_to_indices(self, decoder_outputs):
        """
        Evaluate on the decoder outputs(logits), find the top 1 indices.
        Please confirm that the model is on evaluation mode if dropout/batch_norm layers have been added
        :param decoder_outputs: the output sequence from decoder, shape = T x B x V
        """
        decoded_indices = []
        batch_size = decoder_outputs.size(1)
        decoder_outputs = decoder_outputs.transpose(0, 1)  # S = B x T x V

        for b in range(batch_size):
            top_ids = self._decode_to_index(decoder_outputs[b])
            decoded_indices.append(top_ids.data[0])
        return decoded_indices