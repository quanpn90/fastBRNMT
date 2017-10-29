from __future__ import division

import sys, tempfile
import nmt
import nmt.modules
#~ from nmt.metrics.gleu import sentence_gleu
#~ from nmt.metrics.sbleu import sentence_bleu
#~ from nmt.metrics.bleu import moses_multi_bleu
from nmt.utils import compute_score
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math

class Evaluator(object):
    
    def __init__(self, model, dataset, srcFile, tgtFile, cuda=False):
        
        # some properties
        self.dataset = dataset
        self.dicts = dataset['dicts']
        
        self.model = model
        
        self.eval_files = (srcFile, tgtFile)
        self.cuda = cuda
        
    def setScore(self, score):
        
        self.score = score
    
    def setCriterion(self, criterion):
        
        self.criterion = criterion
    
    
    # Compute perplexity of a data given the model
    def eval_perplexity(self, data, criterion):
        total_loss = 0
        total_words = 0
        
        model = self.model

        model.eval()
        for i in range(len(data)):
            # exclude original indices
            batch = data[i][:-1]
            outputs , _ = model(batch)
            # exclude <s> from targets
            targets = batch[1][1:]
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = targets.view(-1)
            loss = criterion(outputs_flat, targets_flat)
            total_loss += loss.data[0]
            total_words += targets.data.ne(nmt.Constants.PAD).sum()

        model.train()
        return total_loss / total_words
        
    #~ def eval_reinforce(self, data, score, verbose=False):
        #~ 
        #~ total_score = 0
        #~ total_sentences = 0
        #~ 
        #~ total_hit = 0
        #~ total_hit_sentences = 0
        #~ total_gleu = 0
        #~ 
        #~ model = self.model
        #~ model.eval()
        #~ tgtDict = self.dicts['tgt']
        #~ srcDict = self.dicts['src']
        #~ 
        #~ for i in range(len(data)):
            #~ batch = data[i][:-1]
            #~ src = batch[0]
            #~ ref = batch[1][1:]
            #~ # we need to sample
            #~ sampled_sequence = model.sample(src, max_length=100, argmax=True)
            #~ batch_size = ref.size(1)
            #~ 
            #~ for idx in xrange(batch_size):
            #~ 
                #~ tgtIds = sampled_sequence.data[:,idx]
                #~ 
                #~ tgtWords = tgtDict.convertTensorToLabels(tgtIds, nmt.Constants.EOS)        
                                #~ 
                #~ refIds = ref.data[:,idx]
                #~ 
                #~ refWords = tgtDict.convertTensorToLabels(refIds, nmt.Constants.EOS)
                #~ 
                #~ # return a single score value
                #~ s = score(refWords, tgtWords)
                #~ 
                #~ if len(s) > 2:
                    #~ gleu = s[1]
                    #~ hit = s[2]
                    #~ 
                    #~ if hit >= 0:
                        #~ total_hit_sentences += 1
                        #~ total_hit += hit
                #~ 
                #~ if verbose:
                    #~ sampledSent = " ".join(tgtWords)
                    #~ refSent = " ".join(refWords)
                    #~ 
                    #~ if s[0] > 0:
                        #~ print "SAMPLE :", sampledSent
                        #~ print "   REF :", refSent
                        #~ print "Score =", s
                    #~ 
                    #~ 
                #~ 
                #~ 
                #~ # bleu is scaled by 100, probably because improvement by .01 is hard ?
                #~ total_score += s[0] * 100 
                #~ 
            #~ total_sentences += batch_size
        #~ 
        #~ if total_hit_sentences > 0:
            #~ average_hit = total_hit / total_hit_sentences
            #~ print("Average HIT : %.2f" % (average_hit * 100))
        #~ 
        #~ average_score = total_score / total_sentences
        #~ model.train()
        #~ return average_score
            #~ 
    #~ 
    #~ # Compute critic loss of a valid data
    #~ def eval_critic(self, data, dicts, score):
        #~ 
        #~ print("* Evaluating the critic")
        #~ 
        #~ total_loss = 0
        #~ total_sentences = 0
        #~ total_sampled_words = 0
        #~ model = self.model
        #~ critic = self.critic
#~ 
        #~ model.eval()
        #~ critic.eval()
        #~ 
        #~ for i in range(len(data)):
            #~ # exclude original indices
            #~ batch = data[i][:-1]
            #~ 
            #~ src = batch[0]
            #~ ref = batch[1][1:]
            #~ 
            #~ rl_actions = model.sample(src, argmax=True)
            #~ 
            #~ critic_input = Variable(rl_actions.data, volatile=True)
            #~ 
            #~ critic_output = critic(src, critic_input)
            #~ 
            #~ # mask: L x B
            #~ seq_mask = rl_actions.data.ne(nmt.Constants.PAD)
            #~ seq_mask = seq_mask.float()
            #~ num_words_sampled = torch.sum(seq_mask)
            #~ total_sampled_words += num_words_sampled
            #~ 
            #~ # reward for samples from stochastic function
            #~ batch_size = src[0].size(1)
            #~ sampled_reward = compute_score(score, rl_actions, ref, dicts, batch_size) 
            #~ 
            #~ 
            #~ # compute loss for the critic
            #~ # first we have to expand the reward
            #~ expanded_reward = sampled_reward.unsqueeze(0).expand_as(seq_mask)
            #~ 
            #~ # compute weighted loss for critic
            #~ reward_variable = Variable(expanded_reward)
            #~ weight_variable = Variable(seq_mask)
            #~ critic_loss = nmt.modules.Loss.weighted_mse_loss(critic_output, reward_variable, weight_variable)
            #~ total_loss += critic_loss.data[0]
            #~ total_sentences += src[0].size(1)
            #~ 
            #~ 
            #~ 
        #~ 
        #~ model.train()    
        #~ critic.train()
        #~ 
        #~ 
        #~ loss = total_loss / total_sampled_words
        #~ return loss
        #~ 
    #~ 
    #~ # Compute translation quality of a data given the model
    #~ def eval_translate(self, beam_size=1, batch_size=16, bpe=True):
        #~ 
        #~ def addone(f):
            #~ for line in f:
                #~ yield line
            #~ yield None
        #~ 
        #~ srcFile = self.eval_files[0]
        #~ tgtFile = self.eval_files[1]
        #~ 
        #~ if len(srcFile) == 0:
            #~ return 0
        #~ print(" * Translating file %s " % srcFile )
        #~ # initialize the translator for beam search
        #~ translator = nmt.InPlaceTranslator(self.model, self.dicts, beam_size=beam_size, 
                                            #~ batch_size=batch_size, 
                                            #~ cuda=self.cuda)
        #~ 
        #~ srcBatch = []
        #~ 
        #~ count = 0
        #~ 
        #~ # we print translations into temp files
        #~ outF = tempfile.NamedTemporaryFile()
        #~ outRef = tempfile.NamedTemporaryFile()
        #~ 
        #~ nLines = len(open(srcFile).readlines())
        #~ 
        #~ inFile = open(srcFile)
        #~ 
        #~ for line in addone(inFile):
            #~ if line is not None:
                #~ srcTokens = line.split()
                #~ srcBatch += [srcTokens]
                #~ if len(srcBatch) < batch_size:
                    #~ continue
            #~ 
            #~ if len(srcBatch) == 0:
                #~ break        
                #~ 
            #~ predBatch, predScore, goldScore = translator.translate(srcBatch)
            #~ 
            #~ for b in range(len(predBatch)):
                #~ count += 1
                #~ decodedSent = " ".join(predBatch[b][0])
                #~ 
                #~ if bpe:
                    #~ decodedSent = decodedSent.replace('@@ ', '')
                #~ 
                #~ outF.write(decodedSent + "\n")
                #~ outF.flush()
                #~ 
                #~ sys.stdout.write("\r* %i/%i Sentences" % (count , nLines))
                #~ sys.stdout.flush()
            #~ 
            #~ srcBatch = []
            #~ 
        #~ print("\nDone")
        #~ refFile = open(tgtFile)
        #~ 
        #~ for line in addone(refFile):
            #~ if line is not None:
                #~ line = line.strip()
                #~ 
                #~ # remove the hit parts:
                #~ ref = line.split(". ; .", 1)[0]
                #~ ref = ref.strip()
                #~ line = ref
                #~ 
                #~ if bpe:
                    #~ line = line.replace('@@ ', '')
                #~ outRef.write(line + "\n")
                #~ outRef.flush()
        #~ 
        #~ # compute bleu using external script
        #~ bleu = moses_multi_bleu(outF.name, outRef.name)
        #~ refFile.close()
        #~ inFile.close()
        #~ outF.close()
        #~ outRef.close()
        #~ # after decoding, switch model back to training mode
        #~ self.model.train()
        #~ self.model.decoder.attn.applyMask(None)
        #~ 
        #~ return bleu
