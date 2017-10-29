from __future__ import division

import math
import torch
from torch.autograd import Variable, Function
import torch.nn.functional as F
import torch.nn as nn

def variable_recurrent_factory(batch_sizes):
    def fac(inner, reverse=False):
        if reverse:
            return VariableRecurrentReverse(batch_sizes, inner)
        else:
            return VariableRecurrent(batch_sizes, inner)
    return fac


def VariableRecurrent(batch_sizes, inner):
    def forward(input, hidden):
        #~ print(input)
        output = []
        input_offset = 0 
        last_batch_size = batch_sizes[0]
        # to store the rnn states 
        hiddens = [] 
        
        # for gru and normal rnn
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
            
        # loop over the input: 
        # the input has already been flattened, and provided with batch sizes for each time step 
        for batch_size in batch_sizes:
            # get the correct input
            step_input = input[input_offset:input_offset + batch_size]
            input_offset += batch_size

            dec = last_batch_size - batch_size
            # if the last batch size is larger -> have to narrow the hidden layer
            if dec > 0:
                # the next time step have lower batch size -> 
                # we have to store the unused hidden states
                hiddens.append(tuple(h[-dec:] for h in hidden))
                hidden = tuple(h[:-dec] for h in hidden)
            last_batch_size = batch_size
            
            # perform the network activity
            if flat_hidden:
                hidden = (inner(step_input, hidden[0]),)
            else:
                #~ print(step_input)
                hidden = inner(step_input, hidden)
            
            # hidden[0] is the hidden, if using lstm then the rest is cell 
            output.append(hidden[0])
        hiddens.append(hidden)
        
        hiddens.reverse()
        
        # concatenate the accumulated hidden states to form the final one
        hidden = tuple(torch.cat(h, 0) for h in zip(*hiddens))
        assert hidden[0].size(0) == batch_sizes[0]
        if flat_hidden:
            hidden = hidden[0]
        output = torch.cat(output, 0)

        return hidden, output

    return forward


def VariableRecurrentReverse(batch_sizes, inner):
    def forward(input, hidden):
        output = []
        input_offset = input.size(0)
        last_batch_size = batch_sizes[-1]
        initial_hidden = hidden
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
            initial_hidden = (initial_hidden,)
        hidden = tuple(h[:batch_sizes[-1]] for h in hidden)
        for batch_size in reversed(batch_sizes):
            inc = batch_size - last_batch_size
            if inc > 0:
                hidden = tuple(torch.cat((h, ih[last_batch_size:batch_size]), 0)
                               for h, ih in zip(hidden, initial_hidden))
            last_batch_size = batch_size
            step_input = input[input_offset - batch_size:input_offset]
            input_offset -= batch_size

            if flat_hidden:
                hidden = (inner(step_input, hidden[0]),)
            else:
                hidden = inner(step_input, hidden)
            output.append(hidden[0])

        output.reverse()
        output = torch.cat(output, 0)
        if flat_hidden:
            hidden = hidden[0]
        return hidden, output

    return forward
    

def StackedRNN(inners, num_layers, dropout=0, train=True):
    
    # inners contain rnn activity (2 for bidirectional)
    num_directions = len(inners) // num_layers
    total_layers = num_layers * num_directions

    def forward(input, hidden):
    
        next_hidden = []

        
        #~ hidden = list(zip(*hidden))

        for i in range(num_layers):
            all_output = []
            for j in xrange(num_directions):
                l = i * num_directions + j
                inner = inners[l]
                hy, output = inner(input, hidden[l])
                next_hidden.append(hy)
                all_output.append(output)
            #~ for j, inner in enumerate(inners):
                #~ l = i * num_directions + j
                #~ hy, output = inner(input, hidden[l])
                #~ next_hidden.append(hy)
                #~ all_output.append(output)
            
            input = torch.cat(all_output, input.dim() - 1)

            if dropout != 0 and i < num_layers - 1:
                input = F.dropout(input, p=dropout, training=train, inplace=False)

        
        # for lstm
        next_h, next_c = zip(*next_hidden)
        next_hidden = (
            torch.cat(next_h, 0).view(total_layers, *next_h[0].size()),
            torch.cat(next_c, 0).view(total_layers, *next_c[0].size())
        )
        
        return next_hidden, input

    return forward

class RecurrentLayer(nn.Module):
    
    def __init__(self, cell, input_size, hidden_size, num_layers=1, dropout=0, batch_first=True, bidirectional=False):
        
        super(RecurrentLayer, self).__init__()
        
        # we only receive input of format T x B x H
        
        cell_size = hidden_size
        
        if bidirectional:
            cell_size = hidden_size / 2
        
        #~ self.cell = cell(input_size, int(cell_size))
        #~ 
        #~ if bidirectional:
            #~ self.rev_cell = cell(input_size, int(cell_size))
            
        self.cells = nn.ModuleList()
        
        input_size_l = input_size
        for l in xrange(num_layers):
            self.cells.append(cell(input_size_l, int(cell_size)))
            input_size_l = hidden_size
        
        if bidirectional:
            input_size_l = input_size
            self.rev_cells = nn.ModuleList()
            for l in xrange(num_layers):
                self.rev_cells.append(cell(input_size_l, int(cell_size)))
                input_size_l = hidden_size
            
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        
    
    # forward function
    # arguments: input tensor, and hiddens
    def forward(self, input, hidden):
        
        assert isinstance(input, tuple), "This module only receives pack_padded_sequence output"
        
        input_tensor = input[0]
        batch_sizes = input[1] 
        
        rec_factory = variable_recurrent_factory(batch_sizes)
        
        #~ if self.bidirectional:
            #~ layer = (rec_factory(self.cell), rec_factory(self.rev_cell, reverse=True))
        #~ else:
            #~ layer = (rec_factory(self.cell),)
            
        layers = list()
        
        for l in xrange(self.num_layers):
            layers.append(rec_factory(self.cells[l]))
            if self.bidirectional:
                layers.append(rec_factory(self.rev_cells[l], reverse=True))
                
            
        
        func = StackedRNN(layers,
                      self.num_layers,
                      dropout=self.dropout,
                      train=self.training)
            
        nexth, output = func(input_tensor, hidden)
        
        return (output, batch_sizes), nexth
            
#~ def AutogradRNN(cell, input_size, hidden_size, num_layers=1, batch_first=False,
                #~ dropout=0, train=True, bidirectional=False, batch_sizes=None,
                #~ dropout_state=None, flat_weight=None):
#~ 
#~ 
    #~ if batch_sizes is None:
        #~ rec_factory = Recurrent
    #~ else:
        #~ rec_factory = variable_recurrent_factory(batch_sizes)
#~ 
    #~ if bidirectional:
        #~ layer = (rec_factory(cell), rec_factory(cell, reverse=True))
    #~ else:
        #~ layer = (rec_factory(cell),)
    #~ 
    #~ # This function applies stacked RNN and dropout 
    #~ func = StackedRNN(layer,
                      #~ num_layers,
                      #~ (mode == 'LSTM'),
                      #~ dropout=dropout,
                      #~ train=train)
#~ 
    #~ def forward(input, hidden):
        #~ if batch_first and batch_sizes is None:
            #~ input = input.transpose(0, 1)
#~ 
        #~ nexth, output = func(input, hidden, weight)
#~ 
        #~ if batch_first and batch_sizes is None:
            #~ output = output.transpose(0, 1)
#~ 
        #~ return output, nexth
#~ 
    #~ return forward
