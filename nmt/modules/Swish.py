import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
	
	def forward(self, x):
		
		y = x * F.sigmoid(x)
		
		return y
		