from __future__ import division

import sys, tempfile
import nmt
from nmt.utils import split_batch
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import random 
import numpy as np


class XETrainer(object):
    
    
    def __init__(self, model, criterion, optim, trainData, validData, evaluator, dicts, opt):
        
        self.model = model
        self.optim = optim
        self.criterion = criterion
        self.trainData = trainData
        self.validData = validData
        
        self.dicts = dicts
        
        # statistics display during training 
        self.stats = nmt.Stats(optim)
        self.curriculum = -1
        self.evaluator = evaluator
        self.opt = opt
        
    
    def trainEpoch(self, epoch, batchOrder=None):
        
        self.model.train()
        nSamples = len(self.trainData)
        self.stats.reset_stats()
        
        if not batchOrder:
            batchOrder = torch.randperm(len(self.trainData))
        total_loss = 0
        total_words = 0
        
        for i in range(nSamples):

            batchIdx = batchOrder[i] 
            # Exclude original indices.
            batch = self.trainData[batchIdx][:-1]
            batch_size = batch[1].size(1)
            # Exclude <s> from targets.
            targets = batch[1][1:]
            num_words = targets.data.ne(nmt.Constants.PAD).sum()
            total_words += num_words
            
            self.model.zero_grad()
            
            outputs , _ = self.model(batch) 
            
            split_targets = batch[1][1:]
                
            flat_outputs = outputs.view(-1, outputs.size(-1))
            flat_targets = split_targets.view(-1)
            
            # Loss is computed by nll criterion
            loss = self.criterion(flat_outputs, flat_targets)
            loss_value = loss.data[0]
            
            norm_value = batch_size
        
            loss.div(norm_value).backward()

            loss_xe = loss_value
            total_loss += loss_xe
            
            # update the weights
            self.optim.step()
            
            self.stats.report_loss_xe += loss_xe
            self.stats.total_loss_xe += loss_xe
            self.stats.total_words_xe += num_words
            self.stats.report_tgt_words_xe += num_words
            self.stats.report_sentences += batch_size
            self.stats.report_src_words += batch[0][1].data.sum()
            self.stats.report_tgt_words += num_words
            
            if i == 0 or (i % self.opt.log_interval == -1 % self.opt.log_interval):
                self.stats.log(i, epoch, nSamples)
                self.stats.reset_stats()
                
            if self.opt.save_every > 0 and i % self.opt.save_every == -1 % self.opt.save_every :
                valid_loss = self.evaluator.eval_perplexity(self.validData, self.criterion)
                valid_ppl = math.exp(min(valid_loss, 100))
                #~ valid_bleu = evaluator.eval_translate(batch_size = opt.eval_batch_size)
                #~ valid_score = evaluator.eval_reinforce(validData, score)
                print('Validation perplexity: %g' % valid_ppl)
                #~ print('Validation BLEU: %.2f' % valid_bleu)
                #~ print('Validation score: %.2f' % valid_score)

                model_state_dict = self.model.state_dict()
                
                
                #  drop a checkpoint
                ep = float(epoch) - 1 + (i + 1) / nSamples
                checkpoint = {
                    'model': model_state_dict,
                    'dicts': self.dicts,
                    'opt': self.opt,
                    'epoch': ep,
                    'iteration' : i,
                    'batchOrder' : batchOrder,
                    'optim': optim,
                }
                
                file_name = '%s_ppl_%.2f_e%.2f.pt'
                print('Writing to ' + file_name % (opt.save_model, valid_ppl, ep))
                torch.save(checkpoint,
                     file_name
                     % (opt.save_model, valid_ppl, ep))
        
        train_loss = total_loss / total_words
        return train_loss
        
    def run(self):
        
        opt = self.opt
        start_time = time.time()
        
        for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
            print('')

            #  (1) train for one epoch on the training set
            train_loss = self.trainEpoch(epoch)
            train_ppl = math.exp(min(train_loss, 100))
            print('Train perplexity: %g' % train_ppl)

            #  (2) evaluate on the validation set
            valid_loss = self.evaluator.eval_perplexity(self.validData, self.criterion)
            valid_ppl = math.exp(min(valid_loss, 100))
            #~ valid_bleu = evaluator.eval_translate(batch_size = opt.eval_batch_size)
            #~ valid_score = evaluator.eval_reinforce(validData, score)
            print('Validation perplexity: %g' % valid_ppl)
            #~ print('Validation BLEU: %.2f' % valid_bleu)
            #~ print('Validat ion score: %.2f' % valid_score)

            #  (3) update the learning rate
            self.optim.updateLearningRate(valid_ppl, epoch)

            model_state_dict =  self.model.state_dict()

            #  (4) drop a checkpoint
            checkpoint = {
                'model': model_state_dict,
                'dicts': self.dicts,
                'opt': opt,
                'epoch': epoch,
                'iteration' : -1,
                'batchOrder' : None,
                'optim': self.optim,
            }
            
                    
            file_name = '%s_ppl_%.2f_e%d.pt'
            print('Writing to ' + file_name % (opt.save_model, valid_ppl, epoch))
            torch.save(checkpoint,
                       file_name % (opt.save_model, valid_ppl, epoch))

        
    
