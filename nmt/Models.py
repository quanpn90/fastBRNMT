import torch, sys
import torch.nn as nn
from torch.autograd import Variable
import nmt.modules
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import random
import rnnlib
from nmt.modules.GlobalAttention import GroupGlobalAttention
from nmt.modules.LSTM import seqLSTM
from nmt.modules.Swish import Swish
from nmt.modules.WeightDrop import WeightDrop
from nmt.modules.WordDrop import embedded_dropout
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, opt, dicts, embeddings, custom = False):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size
        input_size = opt.word_vec_size
        dropout_value = opt.dropout
        super(Encoder, self).__init__()
        self.word_lut = embeddings
        self.custom = custom
        self.rnn_type = opt.rnn_type
        self.word_dropout = opt.word_dropout
        if self.rnn_type == 'lstm':
            self.rnn = seqLSTM(input_size, self.hidden_size, num_layers=opt.layers, dropout=opt.dropout, rnn_dropout=opt.rnn_dropout, bidirectional=opt.brnn)
        else:
            self.rnn = rnnlib.SRU.SRU(input_size, self.hidden_size, num_layers=opt.layers, dropout=dropout_value, rnn_dropout=opt.rnn_dropout, bidirectional=opt.brnn)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def init_hidden(self, emb, batch_size):
        hidden = []
        for i in xrange(self.layers):
            for j in xrange(self.num_directions):
                h = Variable(emb.data.new(batch_size, self.hidden_size).zero_())
                c = Variable(emb.data.new(batch_size, self.hidden_size).zero_())
                hidden.append((h, c))

        return hidden

    def _fix_enc_hidden(self, h):
        if self.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)).transpose(1, 2).contiguous().view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def _split_hidden(self, h):
        h_states = torch.split(h, 1)
        return h_states

    def forward(self, input, hidden = None):
        input = input[0]
        if isinstance(input, tuple):
            lengths = input[1].data.view(-1).tolist()
            batch_size = input[0].size(1)
            emb = pack(self.word_lut(input[0]), lengths)
            emb_data = emb[0]
        else:
            batch_size = input.size(1)
            emb = self.word_lut(input)
            emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
            emb_data = emb
        outputs, hidden_t = self.rnn(emb, hidden)
        return (hidden_t, outputs)


class Decoder(nn.Module):

    def __init__(self, opt, dicts, embeddings, custom = False):
        self.layers = opt.layers
        input_size = opt.word_vec_size
        self.hidden_size = opt.rnn_size
        self.input_size = input_size
        self.rnn_type = opt.rnn_type
        self.dropout = opt.dropout
        self.rnn_dropout = opt.rnn_dropout
        self.word_dropout = opt.word_dropout
        self.output_size = opt.word_vec_size
        self.word_vec_size = opt.word_vec_size
        super(Decoder, self).__init__()
        self.word_lut = embeddings
        self.rnn_list = nn.ModuleList()
        self.attns = nn.ModuleList()
        if self.rnn_type == 'sru':
            self.rnn = rnnlib.SRU.SRU(input_size, self.hidden_size, num_layers=opt.layers, dropout=opt.dropout, rnn_dropout=opt.rnn_dropout, use_tanh=0, use_relu=1)
        elif self.rnn_type == 'lstm':
            for i in range(self.layers):
                if i == 0:
                    isize = input_size
                else:
                    isize = 2 * self.hidden_size
                rnn = nn.LSTM(isize, self.hidden_size)
                rnn = WeightDrop(rnn, ['weight_hh_l0'], dropout=self.rnn_dropout)
                self.rnn_list.append(rnn)
                attn = GroupGlobalAttention(opt.rnn_size)
                self.attns.append(attn)

        self.dropout_value = opt.dropout
        self.gen_rnn = nn.GRU(self.hidden_size * 2, self.output_size)
        self.swish = Swish()

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden, context):
        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        input_L = emb
        bsize = context.size(1)
        all_attn_scores = []
        context = context.transpose(0, 1).contiguous()
        hidden_out = []
        for l in xrange(self.layers):
            prev_hidden = None
            if hidden is not None:
                prev_hidden = hidden[l]
            rnn_output, hidden_l = self.rnn_list[l](input_L, prev_hidden)
            hidden_out.append(hidden_out)
            attn_context, attn_scores = self.attns[l](rnn_output, context)
            all_attn_scores.append(attn_scores)
            input_L = torch.cat((attn_context, rnn_output), 2)
            if self.training:
                output_mask = Variable(input_L.data.new(bsize, 2 * self.hidden_size).bernoulli_(1 - self.dropout_value).div_(1 - self.dropout_value), requires_grad=False)
                output_mask = output_mask.unsqueeze(0).expand_as(input_L)
                input_L = input_L * output_mask

        output = input_L
        prev_hidden = None
        if hidden is not None and len(hidden) == self.layers + 1:
            prev_hidden = hidden[-1]
        output, prev_hidden = self.gen_rnn(output, prev_hidden)
        hidden_out.append(prev_hidden)
        if self.training:
            output_mask = Variable(output.data.new(bsize, self.hidden_size).bernoulli_(1 - self.dropout_value).div_(1 - self.dropout_value), requires_grad=False)
            output_mask = output_mask.unsqueeze(0).expand_as(output)
            output = output * output_mask
        hidden = hidden_out
        return (output, hidden, all_attn_scores)


class RecurrentEncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, generator):
        super(RecurrentEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def make_init_input(self, src, volatile = False):
        if isinstance(src, tuple):
            src = src[0]
        batch_size = src.size(1)
        i_size = (1, batch_size)
        input_vector = src.data.new(*i_size).fill_(nmt.Constants.BOS)
        return Variable(input_vector, requires_grad=False, volatile=volatile)

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def init_weights(self, init_range):
        pass


class NMTModel(RecurrentEncoderDecoder):

    def __init__(self, encoder, decoder, generator):
        super(NMTModel, self).__init__(encoder, decoder, generator)
        self.saved_actions = []
        self.rewards = []

    def tie_weights(self):
        self.decoder.word_lut.weight = self.generator.net[0].weight

    def tie_join_embeddings(self):
        self.encoder.word_lut.weight = self.decoder.word_lut.weight

    def forward(self, input, timestep_group = 8):
        src = input[0]
        tgt = input[1][:-1]
        enc_hidden, context = self.encoder(src)
        outputs = []
        hidden = None
        batch_size = tgt.size(1)
        length = tgt.size(0)
        init_output = self.make_init_decoder_output(context).unsqueeze(0)
        hiddens, hidden, _attn = self.decoder(tgt, hidden, context, init_output)
        hiddens_split = torch.split(hiddens, timestep_group)
        outputs = []
        for i, hidden_group in enumerate(hiddens_split):
            n_steps = hidden_group.size(0)
            hidden_group = hidden_group.view(-1, hidden_group.size(2))
            output_group = self.generator(hidden_group)
            output_group = output_group.view(n_steps, -1, output_group.size(1))
            outputs.append(output_group)

        outputs = torch.cat(outputs, 0)
        return (outputs, hiddens)

    def init_weights(self, init_range):
        print 'Initializing parameters'
        self.encoder.word_lut.weight.data.uniform_(-init_range, init_range)
        self.decoder.word_lut.weight.data.uniform_(-init_range, init_range)
        self.generator.net[0].bias.data.fill_(0)
        self.generator.net[0].weight.data.uniform_(-init_range, init_range)


class Generator(nn.Module):

    def __init__(self, inputSize, dicts):
        super(Generator, self).__init__()
        self.inputSize = inputSize
        self.outputSize = dicts.size()
        self.net = nn.Sequential(nn.Linear(inputSize, self.outputSize), nn.LogSoftmax(dim=1))

    def forward(self, input):
        return self.net(input)

