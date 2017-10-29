from __future__ import division

import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy

import nmt

def compute_score(score, samples, ref, dicts, batch_size, average=True):
        
    # probably faster than gpu ?
    #~ samples = samples.cpu()
    
    sdata = samples.data.cpu().t().tolist()
    rdata = ref.data.cpu().t().tolist()
    
    tgtDict = dicts['tgt']
    
    s = torch.Tensor(batch_size)
    
    for i in xrange(batch_size):
        
        #~ sampledIDs = sdata[:,i]
        #~ refIDs = rdata[:,i]
        sampledIDs = sdata[i]
        refIDs = rdata[i]
        
        #~ sampledWords = tgtDict.convertTensorToLabels(sampledIDs, nmt.Constants.EOS)
        #~ refWords = tgtDict.convertTensorToLabels(refIDs, nmt.Constants.EOS)
        sampledWords = tgtDict.convertToLabels(sampledIDs, nmt.Constants.EOS)
        refWords = tgtDict.convertToLabels(refIDs, nmt.Constants.EOS)
        
        # note: the score function returns a tuple 
        s[i] = score(refWords, sampledWords)[0]
        #~ assert(len(sampledWords) == lengths[i])
        
    s = s.cuda()
        
    return s


# split the mini-batch (large) into smaller ones
def split_batch(batch, split_size):
    
    splitted_batches = []
    
    src = batch[0][0].data
    tgt = batch[1].data
    length = batch[0][1].data
    volatile = batch[1].volatile
    
    src_splits = torch.split(src, split_size, dim=1)
    tgt_splits = torch.split(tgt, split_size, dim=1)
    length_splits = torch.split(length, split_size, dim=1)
    
    for i, (src_new, tgt_new, length_new) in enumerate(zip(src_splits, tgt_splits, length_splits)):
        
        src_var = Variable(src_new.contiguous(), volatile=volatile)
        tgt_var = Variable(tgt_new.contiguous(), volatile=volatile)
        length_var = Variable(length_new, volatile=volatile)
        
        batch_split = ( (src_var, length_var), tgt_var )
        
        splitted_batches.append(batch_split)
    
    return splitted_batches

"""
Creating embedding lookup tables to be re-used between models
Require dicts and options
"""
def createEmbeddings(opt, dicts):
    
    src_embeddings = nn.Embedding(dicts['src'].size(),
                                     opt.word_vec_size,
                                     padding_idx=nmt.Constants.PAD)
    tgt_embeddings = nn.Embedding(dicts['tgt'].size(),
                                     opt.word_vec_size,
                                     padding_idx=nmt.Constants.PAD)
                                     
    return (src_embeddings, tgt_embeddings)
    
def createNMT(opt, dicts, embeddings, custom=False):
    
    if opt.encoder_type == "text":
        encoder = nmt.Models.Encoder(opt, dicts['src'], embeddings[0], custom=custom)
    elif opt.encoder_type == "img":
        encoder = nmt.modules.ImageEncoder(opt)
    else:
        print("Unsupported encoder type %s" % (opt.encoder_type))
    
    
    decoder = nmt.Models.Decoder(opt, dicts['tgt'], embeddings[1])
    
    # generator: from decoder hidden state to generate words
    generator = nmt.Models.Generator(opt.word_vec_size, dicts['tgt'])
    
    
    model = nmt.Models.NMTModel(encoder, decoder, generator)
    
    return model


#~ def createCritic(opt, dicts, embeddings):
    #~ 
    #~ critic_opt = copy.deepcopy(opt)
    #~ # disable dropout on critic
    critic_opt.dropout = 0.5
    #~ 
    #~ encoder = nmt.Models.Encoder(critic_opt, dicts['src'], embeddings[0])
    #~ 
    #~ decoder = nmt.Models.Decoder(critic_opt, dicts['tgt'], embeddings[1])
    #~ 
    #~ generator = nmt.Models.CriticGenerator(opt.rnn_size)
    #~ 
    #~ critic = nmt.Models.CriticModel(encoder, decoder, generator)
    #~ 
    #~ return critic
