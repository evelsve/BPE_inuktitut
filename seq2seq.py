"""
we train the model sentence by sentence, i.e., setting the batch_size = 1
"""

from __future__ import unicode_literals, print_function, division

import argparse
import logging
import random
import time
from io import open

import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# we are forcing the use of cpu, if you have access to a gpu, you can set the flag to "cuda"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1
MAX_LENGTH = 15
teacher_forcing_ratio = 0.5


class Vocab:
    """ This class handles the mapping between the words and their indicies
    """

    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

######################################################################


def split_lines(input_file):
    """split a file like:
    first src sentence|||first tgt sentence
    second src sentence|||second tgt sentence
    into a list of things like
    [("first src sentence", "first tgt sentence"), 
     ("second src sentence", "second tgt sentence")]
    """
    logging.info("Reading lines of %s...", input_file)
    # Read the file and split into lines
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs
    pairs = [l.split('|||') for l in lines]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    """ Creates the vocabs for each of the langues based on the training corpus.
    """
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab


######################################################################

def tensor_from_sentence(vocab, sentence):
    """creates a tensor from a raw sentence
    """
    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(src_vocab, tgt_vocab, pair):
    """creates a tensor from a raw sentence pair
    """
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor


######################################################################


class EncoderRNN(nn.Module):
    """the class for the enoder RNN"""

    def __init__(self, input_size, hidden_size):
        # input_size: src_side vocabulary size
        # hidden_size: hidden state dimension
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        """runs the forward pass of the encoder
        returns the output and the hidden state
        """
        # TODO 2: complete the forward computation, given the input and the previous hidden state
        #  return the output and the hidden state
        emb = self.embed(input).reshape(1, 1, -1)
        output, hidden = self.gru(emb, hidden)
        return output, hidden

    def get_initial_hidden_state(self):
        # NOTE: you need to change here if you use LSTM as the rnn unit
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    """the class for the decoder 
    """

    def __init__(self, hidden_size, output_size):
        # hidden_size: hidden state dimension
        # output_size: trg_side vocabulary size
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed = nn.Embedding(hidden_size, hidden_size)
        self.outputLayer = nn.Linear(hidden_size, output_size)
        self.soft = nn.LogSoftmax( dim=-1)
        self.gru = nn.GRU(hidden_size, hidden_size)


        # TODO 3: Initilize your word embedding, decoder rnn, output layer, softmax layer
        #  similar to the NLM model in Assignment3
        # done.


    def forward(self, input, hidden):
        """runs the forward pass of the decoder
        returns the log_softmax, hidden state
        """
        emb = self.embed(input)
        emb = F.tanh(emb)
        output, hidden = self.gru(emb, hidden)
        output = self.outputLayer(output[0])
        log_softmax = self.soft(output)
        # TODO 4: complete the forward computation, given the input and the previous hidden state
        # return the following variables
        # log_softmax: the output after applying LogSoftmax function
        # and hidden: hidden states
        # similar to TODO 2, difference: compute the prob over target-side vocabulary given the output
        # done.
        return log_softmax, hidden

    def get_initial_hidden_state(self):
        # NOTE: you need to change here if you use LSTM as the rnn unit
        return torch.zeros(1, 1, self.hidden_size, device=device)


######################################################################

def train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion):
    encoder_hidden = encoder.get_initial_hidden_state()
    # make sure the encoder and decoder are in training mode so dropout is applied
    encoder.train()
    decoder.train()
    optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0
    # encoder-side forward computation
    for ei in range(input_length):
        # TODO 5: feed each input to the encoder, and get the output
        output, enc_hidden = decoder(input_tensor, target_tensor)


    #  set the first input to the decoder is the symbol "SOS"
    decoder_input = torch.tensor([[SOS_index]], device=device)
    # TODO 5: initialize the decoder with the last encoder hidden state
    
    
    # dec_out, dec_hidden = decoder(decoder_input, enc_hidden)
    decoder_hidden = enc_hidden
    # done.


    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # target-side generation
    for di in range(target_length):
        # TODO 5: get the output of the decoder, for each step
        decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden)
        loss += criterion(decoder_out, target_tensor[di])


        # TODO 5: compute the loss
        # done.


        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            decoder_input = target_tensor[di]  # Teacher forcing
        else:
            # Without teacher forcing: use its own predictions as the next input
            topv, topi = decoder_out.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            if decoder_input.item() == EOS_token:
                break

    #  back-propagation step, torch helps you do it automatically
    loss.backward()
    #  update parameters, the optimizer will help you automatically
    optimizer.step()

    loss = loss.item() / target_length  # average of all the steps
    return loss

######################################################################


def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    """ runs translation, returns the output """

    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.get_initial_hidden_state()

        for ei in range(input_length):
            # TODO 6: feed each input to the encoder, and get the output
            enc_out, encoder_hidden = encoder(input_tensor, encoder_hidden)


        #  set the first input to the decoder is the symbol "SOS"
        decoder_input = torch.tensor([[SOS_index]], device=device)
        # TODO 6: initialize the decoder with the last encoder hidden state
        decoder_hidden = encoder_hidden
        # done.

        decoded_words = []

        for di in range(max_length):
            # TODO 6: get the output of the decoder, for each step
            decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden)

            # done.

            topv, topi = decoder_out.data.topk(1)
            if topi.item() == EOS_index:
                decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(tgt_vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

######################################################################


# Translate (dev/test)set takes in a list of sentences and writes out their translates
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH):
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
#

def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################

def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())


######################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=256, type=int,
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=100000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=5000, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--status_every', default=500, type=int,
                    help='print how many examples have been learned ')
    ap.add_argument('--checkpoint_every', default=10000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.001, type=int,
                    help='initial learning rate')
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code, e.g. "fr"')
    ap.add_argument('--tgt_lang', default='en',
                    help='Source (input) language code, e.g. "en"')
    ap.add_argument('--train_file', default='data/fren.train.bpe',
                    help='training file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                    help='dev file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--test_file', default='data/fren.test.bpe',
                    help='test file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence' +
                         ' (for test, target is ignored)')
    ap.add_argument('--out_file', default='out.txt',
                    help='output file for test translations')
    ap.add_argument('--load_checkpoint', nargs=1,
                    help='checkpoint file to start from')
    ap.add_argument('--inference', action='store_true')

    args = ap.parse_args()

    # process the training, dev, test files

    # Create vocab from training data, or load if checkpointed
    # also set iteration 
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0])
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.train_file)

    # TODO 0: initialize the encoder and the decoder here
    # done.
    encoder = EncoderRNN(len(args.src_lang), args.hidden_size)
    decoder = DecoderRNN(len(args.src_lang), args.hidden_size)

    # encoder/decoder weights are randomly initilized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # read in datafiles
    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    if args.load_checkpoint is not None and args.inference:
        translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab)

        references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
        candidates = [clean(sent).split() for sent in translated_sentences]
        test_bleu = corpus_bleu(references, candidates)
        logging.info('Test BLEU score: %.2f', test_bleu)
        return

    # set up optimization/loss
    params = list(encoder.parameters()) + list(decoder.parameters())  # .parameters() returns generator
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()

    # optimizer may have state
    # if checkpointed, load saved state
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every

    # start training
    while iter_num < args.n_iters:
        iter_num += 1
        training_pair = tensors_from_pair(src_vocab, tgt_vocab, random.choice(train_pairs))
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, optimizer, criterion)
        print_loss_total += loss

        if iter_num % args.status_every == 0:
            logging.info('has learnt %d examples', iter_num)
        if iter_num % args.checkpoint_every == 0:
            state = {'iter_num': iter_num,
                     'enc_state': encoder.state_dict(),
                     'dec_state': decoder.state_dict(),
                     'opt_state': optimizer.state_dict(),
                     'src_vocab': src_vocab,
                     'tgt_vocab': tgt_vocab,
                     }
            filename = 'state_%010d.pt' % iter_num
            torch.save(state, filename)
            logging.debug('wrote checkpoint to %s', filename)

        if iter_num % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info('time since start:%s (iter:%d iter/n_iters:%d%%) loss_avg:%.4f',
                         time.time() - start,
                         iter_num,
                         iter_num / args.n_iters * 100,
                         print_loss_avg)
            # translate from the dev set
            translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, n=2)
            translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab)

            references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
            candidates = [clean(sent).split() for sent in translated_sentences]
            dev_bleu = corpus_bleu(references, candidates)
            logging.info('Dev BLEU score: %.2f', dev_bleu)

if __name__ == '__main__':
    main()
