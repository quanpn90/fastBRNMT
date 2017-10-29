import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from nmt.modules.WeightDrop import WeightDrop



# Wrap LSTM in one single module
# Supporting Variational Vertical Dropout and DropConnect 
class seqLSTM(nn.Module):

	def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, 
				 rnn_dropout=0.0, bidirectional=False):
	
		super(seqLSTM, self).__init__()
		self.dropout = dropout
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.bidirectional = bidirectional
		self.rnn_dropout = rnn_dropout
		self.rnn_layers = nn.ModuleList()
		
		for l in xrange(self.num_layers):
			
			if l == 0:
				isize = input_size
			else:
				isize = self.hidden_size
			
			if self.bidirectional:
				hsize = self.hidden_size / 2
			else:
				hsize = self.hidden_size
			
			rnn = nn.LSTM(isize, hsize, bidirectional=self.bidirectional)
			
			rnn = WeightDrop(rnn, ['weight_hh_l0'], dropout=self.rnn_dropout)
			
			self.rnn_layers.append(rnn)
			
	
	def forward(self, input, hidden):
	
		
		input_L = input
		
		hidden_out = []
		
		for L in xrange(self.num_layers):
			
			if hidden is not None:
				prev_hidden_L = hidden[L]
			else:
				prev_hidden_L = None
			
			output_L, hidden_L = self.rnn_layers[L](input_L, prev_hidden_L)
			
			hidden_out.append(hidden_L)
			
			input_L = output_L 
			
			# apply dropout
			if L + 1 < self.num_layers and self.training:
				size = input_L.size()
				# size[1] is dropout, size[2] is hidden size
				output_mask = Variable(input_L.data.new(size[1], size[2]).bernoulli_(1-self.dropout).div_(1-self.dropout), requires_grad=False )
				
				# repeat the mask over the time dimension
				output_mask = output_mask.unsqueeze(0).expand_as(input_L)
					
				input_L = input_L * output_mask
			
		output = input_L
		
		return output, hidden_out