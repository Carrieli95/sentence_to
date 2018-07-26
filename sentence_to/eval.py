from sentence_to.ABD_sts_encoder import VanillaEncoder
from sentence_to.ABD_sts_decoder import VanillaDecoder
from sentence_to.ABD_seq2seq import Seq2Seq
# from sentence_to.sts_data_setup import DataTransformer
from sentence_to.ABD_diction import DataTransformer
from sentence_to.ABD_train_seq import Trainer
from sentence_to import ABD_config
from server_request_file import read_from_url


def main():
    datapath = read_from_url('google-10000-english.txt')
    data_transformer = DataTransformer(datapath, use_cuda=False)

    vanilla_encoder = VanillaEncoder(vocab_size=data_transformer.vocab_size,
                                     embedding_size=ABD_config.encoder_embedding_size,
                                     output_size=ABD_config.encoder_output_size)

    vanilla_decoder = VanillaDecoder(hidden_size=ABD_config.decoder_hidden_size,
                                     output_size=data_transformer.vocab_size,
                                     max_length=data_transformer.max_length,
                                     learning_ratio=ABD_config.teacher_forcing_ratio,
                                     sos_id=data_transformer.SOS_ID,
                                     use_cuda=ABD_config.use_cuda)
    if ABD_config.use_cuda:
        vanilla_encoder = vanilla_encoder.cuda()
        vanilla_decoder = vanilla_decoder.cuda()

    seq2seq = Seq2Seq(encoder=vanilla_encoder,
                      decoder=vanilla_decoder)

    trainer = Trainer(seq2seq, data_transformer, ABD_config.learning_rate, ABD_config.use_cuda)
    trainer.load_model()

    while(True):
        testing_word = input('You say: ')
        if testing_word == "exit":
            break
        results = trainer.evaluate(testing_word)
        print("Model says: %s" % results[0])


if __name__ == "__main__":
    main()