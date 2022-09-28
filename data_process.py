
import numpy as np
import torch

class CMpairLoader(object):
    """Construct easily mixed labels in the same batch, and the remaining samples are randomly placed in subsequent batches
    
    """
    def __init__(self, y_train,data,error_pairs,truetypes,batch_size, mode='train', use_bert=False, raw_text=None):
        """
        pair_onehot = (trueclass,predclass)
        """
        self.use_bert = use_bert
        if self.use_bert:
            self.inp = list(raw_text)
        else:
            self.inp = data[0]
        self.tgt = data[1]
        self.batch_size = batch_size
        self.n_samples = len(data[0])
        self.n_classes = self.tgt.shape[1] 
        self.error_pairs = error_pairs
        self.truetypes = truetypes
        self.y_train = y_train
        self.mode = mode
        self._shuffle_indices()

    def _shuffle_indices(self):
        select_index = []
        for _t,_p in self.error_pairs:
            t_idx = self.y_train[self.y_train.isin([_t])].index
            p_idx = self.y_train[self.y_train.isin([_p])].index
            n_pair_batch = (len(t_idx) + len(p_idx)) // self.batch_size
            t_idx = t_idx.values.tolist()
            p_idx = p_idx.values.tolist()
            np.random.shuffle(t_idx)
            np.random.shuffle(p_idx)
            for i in range(n_pair_batch):
                start = i * self.batch_size //2
                end = (i+1) * self.batch_size //2
                add_index = t_idx[start:end] + p_idx[start:end]
                if len(add_index) != self.batch_size:
                    break
                np.random.shuffle(add_index)
                select_index += add_index
        rt_idx = self.y_train[self.y_train.isin(self.truetypes)].index
        true_index = rt_idx.values.tolist()
        np.random.shuffle(true_index)
        select_index += true_index
        self.indices = np.array(select_index)

        self.n_samples = self.indices.shape[0]
        self.n_batches = self.n_samples // self.batch_size
        if self.n_samples <= self.batch_size:
            self.n_batches +=1
            self.batch_size = self.n_samples
        if self.n_samples % self.batch_size > 0:
            self.n_batches += 1
        self.index = 0
        self.batch_index = 0
        print("n_samples = ",self.n_samples)
        print("n_batch = ",self.n_batches)

    def _create_batch(self):
        batch = []
        n = 0
        while n < self.batch_size and self.index < self.n_samples:
            _index = self.indices[self.index]
            batch.append((self.inp[_index],self.tgt[_index]))
            self.index += 1
            n += 1
        self.batch_index += 1
        seq, label = tuple(zip(*batch))
        if not self.use_bert:
            seq = torch.LongTensor(seq)
        if self.mode not in ['test','augment']:
            label = torch.FloatTensor(label)
        elif self.mode == 'augment':
            label = torch.LongTensor(label)
        return seq, label

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            if self.batch_index == self.n_batches:
                raise StopIteration()
            yield self._create_batch()

class DataLoader(object):
    def __init__(self, data, batch_size, mode='train', use_bert=False, raw_text=None):
        self.use_bert = use_bert
        if self.use_bert:
            self.inp = list(raw_text)
        else:
            self.inp = data[0]
        self.tgt = data[1]
        self.batch_size = batch_size
        self.n_samples = len(data[0])

        self.n_batches = self.n_samples // self.batch_size
        if self.n_samples < self.batch_size:
            self.n_batches +=1
            self.batch_size = self.n_samples
        if self.n_samples % self.batch_size > 0:
            self.n_batches += 1
        self.mode = mode
        self._shuffle_indices()

    def _shuffle_indices(self):
        if self.mode == 'test':
            self.indices = np.arange(self.n_samples)
        else:
            self.indices = np.random.permutation(self.n_samples)
        self.index = 0
        self.batch_index = 0

    def _create_batch(self):
        batch = []
        n = 0
        
        while n < self.batch_size and self.index < self.n_samples:
            _index = self.indices[self.index]
            batch.append((self.inp[_index],self.tgt[_index]))
            self.index += 1
            n += 1
        # print("batch = ",len(batch))
        self.batch_index += 1
        seq, label = tuple(zip(*batch))
        if not self.use_bert:
            seq = torch.LongTensor(seq)
        if self.mode not in ['test','augment']:
            label = torch.FloatTensor(label)
        elif self.mode == 'augment':
            label = torch.LongTensor(label)
        return seq, label

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            if self.batch_index == self.n_batches:
                raise StopIteration()
            yield self._create_batch()