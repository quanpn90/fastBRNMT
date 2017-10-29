from __future__ import division

import math
import torch
from torch.autograd import Variable

import nmt

class Dataset(object):
    def __init__(self, srcData, tgtData, batchSize, cuda,
                 volatile=False, data_type="text", balance=True):
        self.src = srcData
        if tgtData:
            self.tgt = tgtData
            assert(len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        self.cuda = cuda
        self.fullSize = len(self.src)
        self._type = "text"

        self.batchSize = batchSize
        #~ self.numBatches = math.ceil(len(self.src)/batchSize)
        self.volatile = volatile
        
        self.balance = balance
        
        if self.balance:
            self.allocateBatch()
        else:
            self.numBatches = math.ceil(len(self.src)/batchSize)

    #~ # This function allocates the mini-batches (grouping sentences with the same size)
    def allocateBatch(self):
            
        # The sentence pairs are sorted by source already (cool)
        self.batches = []
        
        cur_batch = []
        cur_batch_length = -99
        
        for i in xrange(self.fullSize):
            cur_length = self.src[i].size(0)
            # if the current batch's length is different
            # the we create 
            if cur_batch_length != cur_length:
                if len(cur_batch) > 0:
                    self.batches.append(cur_batch)
                cur_batch_length = cur_length
                cur_batch = []
                
            cur_batch.append(i)
            
            if len(cur_batch) == self.batchSize:
                self.batches.append(cur_batch)
                cur_batch = []
            
        # catch the last batch
        if len(cur_batch) > 0:
            self.batches.append(cur_batch)
        
        self.numBatches = len(self.batches)
                
    def _batchify(self, data, align_right=False,
                  include_lengths=False, dtype="text"):
        
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(nmt.Constants.PAD)
        for i in range(len(data)):
                data_length = data[i].size(0)
                offset = max_length - data_length if align_right else 0
                out[i].narrow(0, offset, data_length).copy_(data[i])
        if include_lengths:
                return out, lengths 
        else:
                return out
              
                
                
    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        
        if self.balance:
            batch = self.batches[index]
            srcData = [self.src[i] for i in batch]
            srcBatch, lengths = self._batchify(
                    srcData,
                    align_right=False, include_lengths=True, dtype=self._type)
    
            if self.tgt:
                tgtData = [self.tgt[i] for i in batch]
                tgtBatch = self._batchify(
                            tgtData,
                            dtype="text")
            else:
                tgtBatch = None
        else:
            srcBatch, lengths = self._batchify(
            self.src[index*self.batchSize:(index+1)*self.batchSize],
            align_right=False, include_lengths=True, dtype=self._type)

            if self.tgt:
                tgtBatch = self._batchify(
                        self.tgt[index*self.batchSize:(index+1)*self.batchSize],
                        dtype="text")
            else:
                tgtBatch = None  

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(srcBatch))
        batch = (zip(indices, srcBatch) if tgtBatch is None
                 else zip(indices, srcBatch, tgtBatch))
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        if tgtBatch is None:
            indices, srcBatch = zip(*batch)
        else:
            indices, srcBatch, tgtBatch = zip(*batch)

        def wrap(b, dtype="text"):
            if b is None:
                return b
            b = torch.stack(b, 0)
            if dtype == "text":
                b = b.t().contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        # wrap lengths in a Variable to properly split it in DataParallel
        lengths = torch.LongTensor(lengths).view(1, -1)
        lengths = Variable(lengths, volatile=self.volatile)
        
        srcTensor = wrap(srcBatch, self._type)
        tgtTensor = wrap(tgtBatch, "text")
        
        
        return (srcTensor, lengths), \
            tgtTensor, indices

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(zip(self.src, self.tgt))
        self.src, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])


        
    
