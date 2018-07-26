import torch
import random

from server_request_file import read_from_url
from sentence_to.ABD_sts_encoder import VanillaEncoder
from sentence_to.ABD_sts_decoder import VanillaDecoder
from sentence_to.ABD_seq2seq import Seq2Seq
# from sentence_to.sts_data_setup import DataTransformer
from sentence_to.ABD_diction import DataTransformer
from sentence_to import ABD_config


class Trainer(object):

    def __init__(self, model, data_transformer, learning_rate, use_cuda,
                 checkpoint_name=ABD_config.checkpoint_name,
                 teacher_forcing_ratio=ABD_config.teacher_forcing_ratio):

        self.model = model

        # record some information about dataset
        self.data_transformer = data_transformer
        self.vocab_size = self.data_transformer.vocab_size
        self.PAD_ID = self.data_transformer.PAD_ID
        self.use_cuda = use_cuda

        # optimizer setting
        self.learning_rate = learning_rate
        self.optimizer= torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.NLLLoss(ignore_index=self.PAD_ID, size_average=True)

        self.checkpoint_name = checkpoint_name

    def train(self, num_epochs, batch_size, pretrained=False):

        if pretrained:
            self.load_model()

        step = 0

        for epoch in range(0, num_epochs):
            mini_batches = self.data_transformer.mini_batches(batch_size=batch_size)
            for input_batch, target_batch in mini_batches:
                self.optimizer.zero_grad()
                decoder_outputs, decoder_hidden = self.model(input_batch, target_batch)

                # calculate the loss and back prop.
                cur_loss = self.get_loss(decoder_outputs, target_batch[0])

                # logging
                step += 1
                if step % 50 == 0:
                    print("Step:", step, "loss of char: ", cur_loss.data[0])
                    self.save_model()
                cur_loss.backward()

                # optimize
                self.optimizer.step()

        self.save_model()

    def masked_nllloss(self):
        # Deprecated in PyTorch 2.0, can be replaced by ignore_index
        # define the masked NLLoss
        weight = torch.ones(self.vocab_size)
        weight[self.PAD_ID] = 0
        if self.use_cuda:
            weight = weight.cuda()
        return torch.nn.NLLLoss(weight=weight).cuda()

    def get_loss(self, decoder_outputs, targets):
        b = decoder_outputs.size(1)
        t = decoder_outputs.size(0)
        targets = targets.contiguous().view(-1)  # S = (B*T)
        decoder_outputs = decoder_outputs.view(b * t, -1)  # S = (B*T) x V
        return self.criterion(decoder_outputs, targets)

    def save_model(self):
        torch.save(self.model.state_dict(), self.checkpoint_name)
        print("Model has been saved as %s.\n" % self.checkpoint_name)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.checkpoint_name))
        print("Pretrained model has been loaded.\n")

    def tensorboard_log(self):
        pass

    def evaluate(self, words):
        # make sure that words is list
        if type(words) is not list:
            words = [words]

        # transform word to index-sequence
        eval_var = self.data_transformer.evaluation_batch(words=words)
        decoded_indices = self.model.evaluation(eval_var)
        results = []
        for indices in decoded_indices:
            results.append(self.data_transformer.vocab.indices_to_sequence(indices))
        return results


def main():
    datapath = read_from_url('google-10000-english.txt')
    data_transformer = DataTransformer(datapath, use_cuda=False)

    # define our models
    vanilla_encoder = VanillaEncoder(vocab_size=data_transformer.vocab_size,
                                     embedding_size=ABD_config.encoder_embedding_size,
                                     output_size=ABD_config.encoder_output_size)

    vanilla_decoder = VanillaDecoder(hidden_size=ABD_config.decoder_hidden_size,
                                     output_size=data_transformer.vocab_size,
                                     max_length=data_transformer.max_length,
                                     sos_id=data_transformer.SOS_ID,
                                     use_cuda=ABD_config.use_cuda,
                                     learning_ratio=ABD_config.teacher_forcing_ratio)
    # if ABD_config.use_cuda:
    #     vanilla_encoder = vanilla_encoder.cuda()
    #     vanilla_decoder = vanilla_decoder.cuda()


    seq2seq = Seq2Seq(encoder=vanilla_encoder,
                      decoder=vanilla_decoder)

    trainer = Trainer(seq2seq, data_transformer, ABD_config.learning_rate, ABD_config.use_cuda)
    trainer.train(num_epochs=ABD_config.num_epochs, batch_size=ABD_config.batch_size, pretrained=False)


if __name__ == "__main__":
    main()