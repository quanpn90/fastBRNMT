#~ """
#~ Global attention takes a matrix and a query vector. It
#~ then computes a parameterized convex combination of the matrix
#~ based on the input query.
#~ 
#~ 
        #~ H_1 H_2 H_3 ... H_n
          #~ q   q   q       q
            #~ |  |   |       |
              #~ \ |   |      /
                      #~ .....
                  #~ \   |  /
                          #~ a
#~ 
#~ Constructs a unit mapping.
    #~ $$(H_1 + H_n, q) => (a)$$
    #~ Where H is of `batch x n x dim` and q is of `batch x dim`.
#~ 
    #~ The full def is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.:
#~ 
#~ """
#~ 
import torch
import torch.nn as nn
from nmt.modules.Swish import Swish

class GroupGlobalAttention(nn.Module):
    """ Arguments: 
        qdim: query dim (decoder hidden state)
        mdim: memory dim (encoder hidden state)
    """
    
    def __init__(self, dim):
        super(GroupGlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=True)
        self.linear_context = nn.Linear(dim, dim, bias=True)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim*2, dim, bias=True)
        self.linear_to_one = nn.Linear(dim, 1, bias=True)
        self.mlp_act = Swish()        
        self.tanh = nn.Tanh()
        self.mask = None
    def applyMask(self, mask):
        self.mask = mask
        
    #~ 
    def forward(self, inputs, context):
        """
        input: targetL x batch x dim
        context: batch x sourceL x dim
        """
        
        bsize = context.size(0)
        src_length = context.size(1)
        tgt_length = inputs.size(0)
        dim = context.size(2)
        
        size = torch.Size([bsize * tgt_length, src_length, dim])
        sizeC = torch.Size([tgt_length, bsize, src_length, dim])
        
        # project the hidden state (query)
        projected_inputs = self.linear_in(inputs.view(bsize * tgt_length, -1)) # tgt_length * batch , dim
        
        projected_inputs = projected_inputs.view(tgt_length, bsize, dim).unsqueeze(2) 
        
        expanded_inputs = projected_inputs.expand(sizeC) # tgt_length *batch x source_length x dim
        
        #~ expanded_inputs = expanded_inputs.view(tgt_length, bsize, src_length, dim)
        
        # project the context
        projected_context = self.linear_context(context.view(bsize * src_length, -1)) # batch * src_length, dim
        
        projected_context = projected_context.view(bsize, src_length, dim).unsqueeze(0)  # 1, batch, src_length, dim
        
        expanded_context = projected_context.expand(sizeC) # tgt_length, bsize, src_length, dim
        
        combined = expanded_inputs + expanded_context
        
        # element wise, no need to reshape
        combined = self.mlp_act(combined)
        
        # project to one dim
        
        combined = combined.view(tgt_length * bsize * src_length, dim)
        
        attn_scores = self.linear_to_one(combined) # * src_length, 1
        
        # TODO: add masking
        attn_scores = attn_scores.squeeze(1).view(bsize * tgt_length, src_length)
        attn_scores = self.sm(attn_scores) # probabilities
        
        attn_scores = attn_scores.view(tgt_length, bsize, src_length) # batch x tgt_length x src_length
        
        # combined_context 
        #~ expanded_context = context.unsqueeze(0).expand(sizeC).contiguous().view(tgt_length*bsize, src_length, dim)
        #~ 
        #~ attn_scores = attn_scores.unsqueeze(2).view(tgt_length*bsize, 1 , src_length)
        #~ 
        #~ attn_context = torch.bmm(attn_scores, expanded_context).squeeze(1)
        #~ 
        #~ attn_context = attn_context.view(tgt_length, bsize, dim)
        
        
        attn_context = torch.bmm(attn_scores.transpose(0,1), context).transpose(0,1) # tgt_length  * batch *  dim
        
        #~ combined_context = torch.cat([inputs, attn_context], 2)
        
        #~ context_output = self.tanh(self.linear_out(combined_context.view(-1, dim*2)))
        
        #~ context_output = context_output.view(tgt_length, bsize, dim)
        
        return attn_context, attn_scores
        
    #~ def forward(self, inputs, context):
        #~ """
        #~ input: targetL x batch x dim
        #~ context: batch x sourceL x dim
        #~ """
        #~ 
        #~ bsize = context.size(0)
        #~ src_length = context.size(1)
        #~ tgt_length = inputs.size(0)
        #~ dim = context.size(2)
        #~ 
        #~ reshaped_ctx = context.contiguous().view(bsize * src_length, dim)
        #~ 
        #~ projected_ctx = self.linear_context(reshaped_ctx)
        #~ 
        #~ projected_ctx = projected_ctx.view(bsize, src_length, dim)
        #~ 
        #~ attn_scores = []
        #~ 
        #~ outputs = []
        #~ 
        #~ # split the inputs
        #~ 
        #~ inputs_split = torch.split(inputs, 1)
        #~ 
        #~ for input_t in inputs_split:
                #~ input = input_t.squeeze(0) # batch * dim
                    #~ 
                #~ # project the hidden state (query)
                #~ targetT = self.linear_in(input).unsqueeze(1)  # batch x 1 x dim
#~ 
                #~ # MLP attention model
#~ 
                #~ repeat = targetT.expand_as(projected_ctx)
                #~ sum_query_ctx = repeat + projected_ctx 
                #~ sum_query_ctx = sum_query_ctx.view(bsize * src_length, dim)
#~ 
                #~ mlp_input = self.mlp_tanh(sum_query_ctx)
                #~ mlp_output = self.linear_to_one(mlp_input)
#~ 
                #~ mlp_output = mlp_output.view(bsize, src_length, 1)
                #~ attn = mlp_output.squeeze(2)
                #~ 
                #~ # Get attention
                #~ attn = self.sm(attn)
                #~ 
                #~ attn_scores.append(attn)
                #~ 
                #~ attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL
#~ 
                #~ weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
                #~ 
                #~ contextCombined = torch.cat((weightedContext, input), 1)
                #~ 
                #~ contextOutput = self.tanh(self.linear_out(contextCombined))
                #~ 
                #~ outputs += [contextOutput]
        #~ 
        #~ outputs = torch.stack(outputs)
#~ 
        #~ 
        #~ return outputs, attn_scores
            
            
        
        #~ size = torch.Size([bsize, tgt_length, src_length, dim])
        #~ 
        #~ # project the hidden state (query)
        #~ projected_inputs = self.linear_in(inputs.view(bsize * tgt_length, -1)) # batch * tgt_length, dim
        #~ 
        #~ projected_inputs = projected_inputs.view(bsize, tgt_length, dim).unsqueeze(2) # batch x tgt_length x 1 x dim
        #~ 
        #~ expanded_inputs = projected_inputs.expand(size) # batch x tgt_length x source_length x dim
        #~ 
        #~ # project the context
        #~ projected_context = self.linear_context(context.view(bsize * src_length, -1)) # batch * src_length, dim
        #~ 
        #~ projected_context = projected_context.view(bsize, src_length, dim).unsqueeze(1)  # batch, 1, src_length, dim
        #~ 
        #~ expanded_context = projected_context.expand(size) # batch x tgt_length x source_length x dim
        #~ 
        #~ combined = expanded_inputs + expanded_context
        #~ 
        #~ # element wise, no need to reshape
        #~ combined = self.mlp_tanh(combined)
        #~ 
        #~ # project to one dim
        #~ 
        #~ combined = combined.view(bsize * tgt_length * src_length, dim)
        #~ 
        #~ attn_scores = self.linear_to_one(combined) # bsize * tgt_length * src_length, 1
        #~ 
        #~ # TODO: add masking
        #~ 
        #~ attn_scores = self.sm(attn_scores) # probabilities
        #~ 
        #~ attn_scores = attn_scores.view(bsize, tgt_length, src_length, 1).squeeze(3) # batch x tgt_length x src_length
        #~ 
        #~ # combined_context 
        #~ attn_context = torch.bmm(attn_scores, context) # batch * tgt_length * dim
        #~ 
        #~ combined_context = torch.cat([inputs, attn_context], 2)
        #~ 
        #~ context_output = self.tanh(self.linear_out(combined_context.view(-1, dim*2)))
        #~ 
        #~ context_output = context_output.view(bsize, tgt_length, dim)
        
        return context_output, attn_scores
        


class GlobalAttention(nn.Module):
    def __init__(self, dim):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_context = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim*2, dim, bias=False)
        self.linear_to_one = nn.Linear(dim, 1, bias=True)
        self.tanh = nn.Tanh()
        self.mlp_tanh = nn.Tanh()
        self.mask = None
        
        # For context gate
        self.linear_cg = nn.Linear(dim*2, dim, bias=True)
        self.sigmoid_cg = nn.Sigmoid()

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        bsize = context.size(0)
        seq_length = context.size(1)
        dim = context.size(2)
        
        # project the hidden state (query)
        targetT = self.linear_in(input).unsqueeze(1)  # batch x 1 x dim
        
        # project the context (keys and values)
        reshaped_ctx = context.contiguous().view(bsize * seq_length, dim)
        
        projected_ctx = self.linear_context(reshaped_ctx)
        
        projected_ctx = projected_ctx.view(bsize, seq_length, dim)
        
        # MLP attention model
        
        repeat = targetT.expand_as(projected_ctx)
        sum_query_ctx = repeat + projected_ctx 
        sum_query_ctx = sum_query_ctx.view(bsize * seq_length, dim)
        
        mlp_input = self.mlp_tanh(sum_query_ctx)
        mlp_output = self.linear_to_one(mlp_input)
        
        mlp_output = mlp_output.view(bsize, seq_length, 1)
        attn = mlp_output.squeeze(2)
        #~ attn = mlp_output.squeeze(1).view(bsize, seq_length)
        #~ attn = mlp_output.view(bsize, seq_length) # batch x sourceL

        # Get attention
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
            self.mask = None
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        contextCombined = torch.cat((weightedContext, input), 1)
        
        #ContextGate
        contextGate = self.sigmoid_cg(self.linear_cg(contextCombined))
        inputGate = 1 - contextGate
        
        gatedContext = weightedContext * contextGate
        gatedInput = input * inputGate
        gatedContextCombined = torch.cat((gatedContext, gatedInput), 1)
        

        contextOutput = self.tanh(self.linear_out(gatedContextCombined))

        return contextOutput, attn



