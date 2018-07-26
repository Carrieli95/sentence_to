from sentence_to.sqs_train import Train
from sentence_to.sts_data_setup import DataTransformer
from sentence_to.sqs_encoder import Encoder
from sentence_to.sqs_decoder import Decoder
from sentence_to.sqs_sts import EncodeAndDecode
from sentence_to.server_request import read_from_url


sentence_list = read_from_url('bj_sentence.txt')[0:1000]
batch_size = 100
embedding_size = 400
encoder_output_size = 400
decoder_hidden_size = encoder_output_size
checkpoint_name = 'auto_encoder_record.pt'
epochs = 10
forcing_ratio = 0.7
use_cuda = False
learning_rate = 1e-3

data_prepare = DataTransformer(sentence_list, use_cuda=use_cuda)

Encoder_pre = Encoder(vocab_size=data_prepare.vocab_size, embedded_size=embedding_size, out_size=encoder_output_size)
Decoder_pre = Decoder(hidden_size=decoder_hidden_size, output_size=data_prepare.vocab_size, max_sentence=data_prepare.max_length,
                      use_cuda=use_cuda, teacher_ratio=forcing_ratio, sos_id=data_prepare.SOS_ID)

seq2seq = EncodeAndDecode(encoder=Encoder_pre, decoder=Decoder_pre)

define_train = Train(model=seq2seq, data_trans=data_prepare, learning_rate=learning_rate, use_cuda=use_cuda,
                     teacher_forcing_ratio=forcing_ratio, checkpoint_name=checkpoint_name)
define_train.trainner(epochs=epochs, batch_size=batch_size, load_model=False)
