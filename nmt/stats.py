from __future__ import division

import math
import torch
from torch.autograd import Variable

import nmt
import time

"""
Statistics class
Recording the stats generated during training
"""
class Stats(object):
    
    def __init__(self, optim):
        self.total_loss_xe = 0
        self.total_words_xe = 0
        self.report_loss_xe = 0
        self.report_tgt_words_xe = 0
        self.report_src_words = 0
        self.report_tgt_words = 0
        self.report_sampled_words = 0
        
        self.report_sentences = 0
        
        self.total_critic_loss = 0
        self.total_sent_reward = 0
        
        self.mode = 'xe'
        self.begin_time = time.time()
        self.optim = optim
        self.start = time.time()
    
    # switching mode between 'xe', 'rf' or 'ac'
    def switch_mode(self, mode):
        self.mode = mode
    
    def reset_time(self):
        self.start = time.time()
        
    def reset_stats(self):
        
        self.report_loss_xe, self.report_tgt_words_xe = 0, 0
        self.report_src_words = 0
        self.report_tgt_words = 0
        self.report_sentences = 0 
        self.total_sent_reward = 0
        self.total_critic_loss = 0
        self.report_sampled_words = 0
        self.report_tgt_words_xe = 0
        self.total_loss_xe = 0

    
    def log(self, iteration, epoch, dataSize):
        
        if self.mode == 'xe':
            
            print(("Epoch %2d, %5d/%5d; ; ppl: %6.2f; lr: %1.6f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, iteration+1, dataSize,
               math.exp(self.report_loss_xe / (self.report_tgt_words_xe + 1e-6)),
               self.optim.getLearningRate(),
               self.report_src_words/(time.time()-self.start),
               self.report_tgt_words/(time.time()-self.start),
               time.time()-self.begin_time))
        
        
        elif self.mode == 'rf':
            print(("Epoch %2d, %5d/%5d; ; reward: %6.2f; lr: %1.6f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, iteration+1, dataSize,
               self.total_sent_reward / (self.report_sentences + 1e-6),
               self.optim.getLearningRate(),
               self.report_src_words/(time.time()-self.start),
               self.report_tgt_words/(time.time()-self.start),
               time.time()-self.begin_time))
        
        elif self.mode == 'ac':
            print(("Epoch %2d, %5d/%5d; ; reward: %6.2f; critic loss: %6.4f; lr: %1.6f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, iteration+1, dataSize,
               self.total_sent_reward / (self.report_sentences),
               self.total_critic_loss / (self.report_sampled_words), 
               self.optim.getLearningRate(),
               self.report_src_words/(time.time()-self.start),
               self.report_tgt_words/(time.time()-self.start),
               time.time()-self.begin_time))
        
        self.reset_time()
