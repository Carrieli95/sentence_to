import torch
from urllib.request import urlopen

# Define hyper parameter
use_cuda = True if torch.cuda.is_available() else False

# for training
num_epochs = 10
# 定义了神经网络权重的更新次数
batch_size = 128
learning_rate = 1e-3

# dataset_path = urlopen('http://192.168.90.243:8000/sentence_to/dataset/bj_sentence.txt')
# dataset_path = urlopen('http://192.168.90.243:8000/sentence_to/dataset/google-10000-english.txt')

# for model
encoder_embedding_size = 256
encoder_output_size = 256
decoder_hidden_size = encoder_output_size
teacher_forcing_ratio = .5
# max_length = 20

# for logging
checkpoint_name = 'auto_encoder.pt'

if __name__ == '__main__':
    response = dataset_path.read().decode("utf-8")
    print(response)
    file = response.split("\n")
    with open(dataset_path, 'r', encoding='utf-8') as dataset:
        for sen in dataset:
            word = sen.strip('\n')
            print(word)