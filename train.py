from __future__ import division

import sys, tempfile
import nmt
import nmt.Markdown
#~ import nmt.modules
#~ from nmt.metrics.gleu import sentence_gleu
#~ from nmt.metrics.sbleu import sentence_bleu
#~ from nmt.metrics.bleu import moses_multi_bleu
#~ from nmt.metrics.hit import HitMetrics
from nmt.trainer.Evaluator import Evaluator
from nmt.trainer.XETrainer import XETrainer 
from nmt.utils import split_batch, compute_score
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import random 
import numpy as np

parser = argparse.ArgumentParser(description='train.py')
nmt.Markdown.add_md_help_argument(parser)

# Data options

parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-train_from_state_dict', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model's state_dict.""")
parser.add_argument('-train_from', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")
parser.add_argument('-balance_batch', default=1, type=int,
                    help="""balance mini batches (same source sentence length)""")
parser.add_argument('-join_vocab', action='store_true',
                    help='Use a bidirectional encoder')
# Model options

parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=512,
                    help='Size of LSTM hidden states')
parser.add_argument('-rnn_type', default='lstm',
                    help='Recurrent cell type')
parser.add_argument('-word_vec_size', type=int, default=512,
                    help='Word embedding sizes')
parser.add_argument('-hidden_output_size', type=int, default=-1,
                    help='Size of the final hidden (output embedding)')
parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
# parser.add_argument('-residual',   action="store_true",
#                     help="Add residual connections between RNN layers.")
parser.add_argument('-brnn', action='store_true',
                    help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")

# Optimization options
parser.add_argument('-encoder_type', default='text',
                    help="Type of encoder to use. Options are [text|img].")
parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-computational_batch_size', type=int, default=-1,
                    help='Maximum batch size for computation. By default it is the same as batch size. But we can split the large minibatch to fit in the GPU.')
parser.add_argument('-max_generator_batches', type=int, default=64,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")                   
parser.add_argument('-eval_batch_size', type=int, default=8,
                    help='Maximum batch size for decoding eval')
parser.add_argument('-tie_weights', action='store_true',
                    help='Tie the weights of the decoder embedding and logistic regression layer')
parser.add_argument('-epochs', type=int, default=13,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-optim', default='sgd',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between RNN stacks.')
parser.add_argument('-rnn_dropout', type=float, default=0.1,
                    help='RNN Dropout probability; applied between RNN hiddens.')
parser.add_argument('-word_dropout', type=float, default=0.1,
                    help='Dropout probability; applied on discrete word types.')
parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-extra_shuffle', action="store_true",
                    help="""By default only shuffle mini-batch order; when true,
                    shuffle and re-assign mini-batches""")
                    
# for reinforcement learning
parser.add_argument('-reinforce_rate', type=float, default=0.0,
                    help='Rate of using reinforcement learning during training')
parser.add_argument('-hit_alpha', type=float, default=0.5,
                    help='Rate of balancing gleu and hit')
parser.add_argument('-reinforce_metrics', default='gleu',
                    help='Metrics for reinforcement learning. Default = gleu')
parser.add_argument('-reinforce_sampling_number', type=int, default=1,
                    help='Number of samples during reinforcement learning')
parser.add_argument('-actor_critic', action='store_true',
                    help='Use actor critic algorithm (default is self-critical)')             
parser.add_argument('-normalize_rewards', action='store_true',
                    help='Normalize the rewards')     
parser.add_argument('-pretrain_critic', action='store_true',
                    help='Pretrain the critic')  
parser.add_argument('-use_entropy', action='store_true',
                    help='Use the entropy term in the A2C loss function)')             
parser.add_argument('-entropy_coeff', type=float, default=0.01,
                    help='Entropy coefficient in the A2C loss function.') 
# learning rate
parser.add_argument('-learning_rate', type=float, default=1.0,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1,
                    adadelta = 1, adam = 0.001""")
parser.add_argument('-learning_rate_decay', type=float, default=1,
                    help="""If update_learning_rate, decay learning rate by
                    this much if (i) perplexity does not decrease on the
                    validation set or (ii) epoch has gone past
                    start_decay_at""")
parser.add_argument('-start_decay_at', type=int, default=1000,
                    help="""Start decaying every epoch after and including this
                    epoch""")
parser.add_argument('-reset_optim', action='store_true',
                    help='Use a bidirectional encoder')

    
# pretrained word vectors

parser.add_argument('-pre_word_vecs_enc',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")

# GPU
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-disable_cudnn', action='store_true',
                    help='Use a bidirectional encoder')
parser.add_argument('-seed', default=9999, nargs='+', type=int,
                    help="Seed for deterministic runs.")

parser.add_argument('-log_interval', type=int, default=100,
                    help="Print stats at this interval.")
parser.add_argument('-save_every', type=int, default=-1,
                    help="Save every this interval.")
parser.add_argument('-sample_every', type=int, default=1e99,
                    help="Save every this interval.")

parser.add_argument('-valid_src', default='',
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', default='',
                    help="Path to the validation target data")                   
opt = parser.parse_args()

if opt.computational_batch_size <= 0 :
    opt.computational_batch_size = opt.batch_size

print(opt)



print '__PYTORCH VERSION:', torch.__version__


if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, should run with -gpus 0")

if opt.gpus:
    cuda.set_device(opt.gpus[0])
    cuda.manual_seed_all(opt.seed)
    
    if not opt.disable_cudnn:
        print '__CUDNN VERSION:', torch.backends.cudnn.version()
    else:
        torch.backends.cudnn.enabled = False
    
    import rnnlib.SRU

torch.manual_seed(opt.seed)

def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[nmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit



def main():
    print("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)
    print("Done")
    dict_checkpoint = (opt.train_from if opt.train_from
                       else opt.train_from_state_dict)
    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        dataset['dicts'] = checkpoint['dicts']

    trainData = nmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, opt.gpus,
                             data_type=dataset.get("type", "text"), balance=(opt.balance_batch==1))
    validData = nmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.eval_batch_size, opt.gpus,
                             volatile=True,
                             data_type=dataset.get("type", "text"))

    dicts = dataset['dicts']
    print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    print(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')
    
    criterion = NMTCriterion(dataset['dicts']['tgt'].size())
    
    embeddings = nmt.utils.createEmbeddings(opt, dicts)

    model = nmt.utils.createNMT(opt, dicts, embeddings)
    print "Neural Machine Translation Model"
    
    print(model)
    
    if opt.train_from_state_dict:
                
        print('Loading model from checkpoint at %s'
              % opt.train_from_state_dict)
        model_state_dict = {k: v for k, v in checkpoint['model'].items() if 'criterion' not in k}
        model.load_state_dict(model_state_dict)
        opt.start_epoch = int(math.floor(checkpoint['epoch'] + 1))
        del checkpoint['model'] 
    
    if not opt.train_from_state_dict and not opt.train_from:
        # initialize parameters for the nmt model
        
        model.init_weights(opt.param_init)
        #~ for p in model.parameters():
            #~ p.data.uniform_(-opt.param_init, opt.param_init)

        model.encoder.load_pretrained_vectors(opt)
        model.decoder.load_pretrained_vectors(opt)

    if len(opt.gpus) >= 1:
        model.cuda()
    else:
        model.cpu()
        
    if opt.tie_weights:
        print("Share weights between decoder input and output embeddings")
        model.tie_weights()   
        
    if opt.join_vocab:
        print("Share weights between source and target embeddings")
        model.tie_join_embeddings()
    
    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
    
    
    if opt.reset_optim or not opt.train_from_state_dict:    
        
        optim = nmt.Optim(
                opt.optim, opt.learning_rate, opt.max_grad_norm,
                lr_decay=opt.learning_rate_decay,
                start_decay_at=opt.start_decay_at
        )
    
    else:
         print('Loading optimizer from checkpoint:')
         optim = checkpoint['optim']  
         # Force change learning rate
         optim.lr = opt.learning_rate
         optim.start_decay_at = opt.start_decay_at
         optim.start_decay = False
         del checkpoint['optim']
        
    optim.set_parameters(model.parameters())
    optim.setLearningRate(opt.learning_rate)
    
    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)
    
    evaluator = Evaluator(model, dataset, opt.valid_src, opt.valid_tgt, cuda=(len(opt.gpus) >= 1))
    
    valid_loss = evaluator.eval_perplexity(validData, criterion)
    valid_ppl = math.exp(min(valid_loss, 100))
    print('* Initial Perplexity : %.2f' % valid_ppl)
    
    print('* Start training ... ')
    
    trainer = XETrainer(model, criterion, optim, trainData, validData, evaluator, dicts, opt)

    trainer.run()

if __name__ == "__main__":
    main()

