# -*- coding: utf-8 -*-
import torch
import numpy as np
from server_request_file import read_from_url

from torch.autograd import Variable


class Vocabulary(object):

    def __init__(self):
        self.char2idx = {'SOS': 0, 'EOS': 1, 'PAD': 2, 'UNK': 3}
        self.idx2char = {0: 'SOS', 1: 'EOS', 2: 'PAD', 3: 'UNK'}
        self.num_chars = 4  # 更改4-5
        self.max_length = 0
        self.word_list = []  # 目前好像我们不用
        self.sentence_list = []
        # self.sentence_long = []
        # self.sentence_max = 0

    def build_vocab(self, sentence_list):
        """建立汉语单字和index之间的对应关系,以单句为考量标准。"""
        for sentence in sentence_list:
            self.sentence_list.append(sentence)
            all_word = self.split_sentence(sentence)
            # self.sentence_long.append(len(all_word))  # 统计一个句子的最大长度，先不考虑开始sos和结尾eos。
            self.word_list.extend(all_word)  # 扩充word_list

            if self.max_length < len(all_word):
                self.max_length = len(all_word)
            for word in all_word:
                if word not in self.char2idx:
                    self.char2idx[word] = self.num_chars
                    self.idx2char[self.num_chars] = word
                    self.num_chars += 1
        # self.sentence_max = max(self.sentence_long)  # 记录最长句长。

    # todo: 考虑到底使用sentence为单位，还是以event为单位。目的是什么

    @staticmethod
    def split_sentence(sentence):
        return list(sentence)

    def sequence_to_indices(self, sentence, add_eos=False, add_sos=False):
        """Transform a char sequence to index sequence
            :param sentence: a string composed with chars
            :param add_eos: if true, add the <EOS> tag at the end of given sentence
            :param add_sos: if true, add the <SOS> tag at the beginning of given sentence
        """
        index_sequence = [self.char2idx['SOS']] if add_sos else []

        for word in self.split_sentence(sentence):
            if word not in self.char2idx:
                index_sequence.append(self.char2idx["UNK"])
            else:
                index_sequence.append(self.char2idx[word])

        if add_eos:
            index_sequence.append(self.char2idx['EOS'])

        return index_sequence

    def indices_to_sequence(self, index_sentence):
        sentence = ""
        for index in index_sentence:
            need_to_add = self.idx2char[index]
            if need_to_add == "EOS":
                break
            else:
                sentence = sentence + need_to_add
        return sentence

    def __str__(self):
        # str = "Vocab information:\n"
        string = "汉字与数字标签的对应信息：\n"
        for idx, char in self.idx2char.items():
            string += "Char: %s Index: %d\n" % (char, idx)
        return string


class DataTransformer(object):

    def __init__(self, sentence_list, use_cuda):
        self.indices_sequences = []
        self.use_cuda = use_cuda

        # Load and build the vocab 根据传入的文本列表数据扩充、建立字典库
        self.vocab = Vocabulary()
        self.vocab.build_vocab(sentence_list)
        self.PAD_ID = self.vocab.char2idx["PAD"]
        self.SOS_ID = self.vocab.char2idx["SOS"]
        self.vocab_size = self.vocab.num_chars
        self.max_length = self.vocab.max_length
        # self.sentence_max = self.vocab.sentence_max
        # 这里也加入了sentence_max 的参数
        self._build_training_set(sentence_list)

    def _build_training_set(self, sentence_list):
        # Change sentences to indices, and append <EOS> at the end of all sentence
        for sentence in sentence_list:  # 这里面，vocab.sentence_list 其实就等于sentence_list
            indices_seq = self.vocab.sequence_to_indices(sentence, add_eos=True)
            target_seq = self.vocab.sequence_to_indices(sentence, add_eos=True, add_sos=True)
            """
            把TARGET_SEQ 改成了以0为开始的tensor变量
            """
            # input and target are the same in auto-encoder
            self.indices_sequences.append([indices_seq, target_seq[:]])  # 这里就可以解决target变换的问题了，通过传入不同的target

    def mini_batches(self, batch_size):
        # input_batches = []
        # target_batches = []

        np.random.shuffle(self.indices_sequences)  # 目前没懂这一步的含义，感觉是否打乱并不重要
        mini_batches = [
            self.indices_sequences[k: k + batch_size]
            for k in range(0, len(self.indices_sequences), batch_size)
        ]  # list 套list，每一句话对应的indices向量

        print('indices_sequence 的长度为%s' % len(self.indices_sequences))
        print(len(mini_batches))

        for batch in mini_batches:
            seq_bags = sorted(batch, key=lambda k: len(k[0]), reverse=True)  # sorted by input_lengths
            input_seqs = [ele[0] for ele in seq_bags]
            target_seqs = [ele[1] for ele in seq_bags]

            input_lengths = [len(s) for s in input_seqs]
            input_max = input_lengths[0]
            input_padded = [self.pad_sequence(s, input_max) for s in input_seqs]

            target_lengths = [len(s) for s in target_seqs]
            target_max = target_lengths[0]
            target_padded = [self.pad_sequence(s, target_max) for s in target_seqs]

            input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)  # time * batch
            target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)  # time * batch

            print(target_var, target_var.size())
            # 将列向量转化成行向量

            # if self.use_cuda:
            #     input_var = input_var.cuda()
            #     target_var = target_var.cuda()

            yield (input_var, input_lengths), (target_var, target_lengths)

    def pad_sequence(self, sequence, max_length):
        """填充sequence_index里面空缺的位置"""
        sequence += [self.PAD_ID for i in range(max_length - len(sequence))]
        return sequence

    def evaluation_batch(self, sentence_list):
        """
        words 之前传入的参数，但应该要在这里变成sentence_list
        Prepare a batch of var for evaluating  用于测试，或者说用于predict预测的
        :param sentence_list: a list, store the testing data
        :return: evaluation_batch
        """
        evaluation_batch = []

        for sentence in sentence_list:
            indices_seq = self.vocab.sequence_to_indices(sentence, add_eos=True)
            evaluation_batch.append([indices_seq])

        seq_bags = sorted(evaluation_batch, key=lambda k: len(k[0]), reverse=True)
        input_seqs = [pair[0] for pair in seq_bags]
        input_lengths = [len(s) for s in seq_bags]
        input_max = input_lengths[0]
        input_padded = [self.pad_sequence(s, input_max) for s in input_seqs]

        input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)  # time * batch

        if self.use_cuda:
            input_var = input_var.cuda()

        return input_var, input_lengths


if __name__ == '__main__':
    vocab = Vocabulary()
    res = read_from_url("bj_sentence.txt")
    vocab.build_vocab(res)

    test = res[0]
    print("输入", test)
    ids = vocab.sequence_to_indices(test)
    print("将对应输入语句转换成数子表示的列表", ids)
    sent = vocab.indices_to_sequence(ids)
    print("输出:", sent)

    data_transformer = DataTransformer(res, use_cuda=False)
    count = 0
    print(data_transformer.vocab_size)
    for i in range(2):
        for ib, tb in data_transformer.mini_batches(batch_size=1):
            f, s = ib
            print('input_var: ', f.size())
            print('input_len: ', s)
            for i in ib:
                count += 1
            break