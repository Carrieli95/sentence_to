from sentence_to import sts_data_setup, sqs_decoder, sqs_encoder
import torch
import torch.nn as nn

# chekpoint_name = 'auto_encoder_record.pt'
# epochs = 10


class Train():

    def __init__(self, model, data_trans, learning_rate, use_cuda,
                 teacher_forcing_ratio, checkpoint_name):
        self.model = model
        self.learn_rate = learning_rate
        self.use_cuda = use_cuda
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.checkpoint_name = checkpoint_name
        self.data_trans = data_trans
        self.PAD_ID = self.data_trans.PAD_ID

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.correction = torch.nn.NLLLoss(ignore_index=self.PAD_ID, size_average=True)

    def trainner(self, batch_size, epochs, load_model=False):
        if load_model:
            self.load_model()

        gradient = 0

        for circles in range(0, epochs):
            all_batches = self.data_trans.mini_batches(batch_size=batch_size)
            for input_batches, target_batches in all_batches:
                self.optimizer.zero_grad()
                decode_outputs, decode_hidden = self.model(input_batches, target_batches)

                curr_loss = self.get_loss(decode_outputs, target_batches[0])

                gradient += 1
                if gradient % 50 == 0:
                    print('step:', gradient, 'loss of char: ', curr_loss.data[0])
                    self.save_model()
                curr_loss.backward()

                self.optimizer.step()
            print('finish the ', circles, ' epochs')
        self.save_model()

    def load_model(self):
        self.model.load_state_dict(torch.load(self.checkpoint_name))
        print("Pretrained model has been loaded.\n")

    def save_model(self):
        torch.save(self.model.state_dict(), self.checkpoint_name)

    def get_loss(self, decode_result, target_result):
        time_len = decode_result.size(0)
        batch_num = decode_result.size(1)
        targets = target_result.contiguous().view(-1)
        decoder_res = decode_result.view(time_len*batch_num, -1)
        # 降维[1*3*4]-->[3*4]  [2*3*4]-->[6*4]
        return self.correction(decoder_res, targets)





