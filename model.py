"""
@File   :  model.py
@Time   :  2022/05/6
@Author  :  Wu Yanan
@Contact :  yanan.wu@bupt.edu.cn
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F

def pair_cosine_similarity(x, x_adv, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    n_adv = x_adv.norm(p=2, dim=1, keepdim=True)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps), (x_adv @ x_adv.t()) / (n_adv * n_adv.t()).clamp(min=eps), (x @ x_adv.t()) / (n * n_adv.t()).clamp(min=eps)

def nt_xent(x, x_adv, mask, cuda=True, t=0.1):
    x, x_adv, x_c = pair_cosine_similarity(x, x_adv)
    x = torch.exp(x / t)    # (batchsize,batchsize)  # sim(xi,xj)
    x_adv = torch.exp(x_adv / t)    # (batchsize,batchsize) # sim(xadvi,xadvj)
    x_c = torch.exp(x_c / t)    # (batchsize,batchsize) # sim(xi,xadvj)
    mask_count = mask.sum(1)    #(batchsize)
    mask_reverse = (~(mask.bool())).long()
    if cuda:
        dis = (x * (mask - torch.eye(x.size(0)).long().cuda()) + x_c * mask) / (x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + mask_reverse
        dis_adv = (x_adv * (mask - torch.eye(x.size(0)).long().cuda()) + x_c.T * mask) / (x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t))) + mask_reverse 
    else:  
        dis = (x * (mask - torch.eye(x.size(0)).long()) + x_c * mask) / (x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + mask_reverse # (batchsize,batchsize)
        dis_adv = (x_adv * (mask - torch.eye(x.size(0)).long()) + x_c.T * mask) / (x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t))) + mask_reverse # (batchsize,batchsize)
    loss = (torch.log(dis).sum(1) + torch.log(dis_adv).sum(1)) / mask_count
    return -loss.mean()


class BiLSTM(nn.Module):
    def __init__(self, embedding_matrix, HIDDEN_DIM, NUM_LAYERS, norm_coef, n_class_seen,  lmcl=True, use_cuda=True, use_bert=False, rcl_cont=False,scl_cont=False):
        super(BiLSTM, self).__init__()
        self.hidden_dim = HIDDEN_DIM
        self.num_layers = NUM_LAYERS
        self.output_dim = n_class_seen
        self.use_bert = use_bert
        self.scl_cont = scl_cont
        self.rcl_cont = rcl_cont
        self.norm_coef = norm_coef
        self.use_cuda = 'cuda' if use_cuda else 'cpu'
        if self.use_bert:
            print('Loading Bert...')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.use_cuda)
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.rnn = nn.GRU(input_size=768, hidden_size=self.hidden_dim,
                              num_layers=self.num_layers,
                              batch_first=True, bidirectional=True).to(self.use_cuda)
            for name, param in self.bert_model.named_parameters():  
                if "encoder.layer.11" in name or "pooler" in name:
                    param.requires_grad = True
        else:
            self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                          _weight=torch.from_numpy(embedding_matrix))
            self.rnn = nn.GRU(input_size=embedding_matrix.shape[1], hidden_size=self.hidden_dim, num_layers=self.num_layers,
                              batch_first=True, bidirectional=True).to(self.use_cuda)
        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim).to(self.use_cuda)
        self.dropout = nn.Dropout(p=0.5)
        self.lmcl = lmcl

    def get_embedding(self, seq):
        seq_embed = self.embedding(seq)
        seq_embed = self.dropout(seq_embed)
        seq_embed = torch.tensor(seq_embed, dtype=torch.float32, requires_grad=True).cuda()
        return seq_embed

    def lmcl_loss(self, probs, label, margin=0.35, scale=30):
        probs = label * (probs - margin) + (1 - label) * probs
        probs = torch.softmax(probs, dim=1)
        return probs

    def forward(self, seq, label=None, mode='ind_pre'):
        if mode in ['ind_pre','finetune']:
            if self.use_bert:
                seq_embed = self.bert_model(**self.bert_tokenizer(seq, return_tensors='pt', padding=True, truncation=True).to(self.use_cuda))[0]
                seq_embed = self.dropout(seq_embed)
                seq_embed = seq_embed.clone().detach().requires_grad_(True).float()
            else:
                seq_embed = self.embedding(seq)
                seq_embed = self.dropout(seq_embed)
                seq_embed = seq_embed.clone().detach().requires_grad_(True).float()
            _, ht = self.rnn(seq_embed)
            ht = torch.cat((ht[0].squeeze(0), ht[1].squeeze(0)), dim=1)
            logits = self.fc(ht)
            if self.lmcl:
                probs = self.lmcl_loss(logits, label)
            else:
                probs = torch.softmax(logits, dim=1)
            ce_loss = torch.sum(torch.mul(-torch.log(probs), label))
            if mode == 'finetune':
                return ce_loss
            elif mode == "ind_pre":
                seq_embed.retain_grad()  # we need to get gradient w.r.t embeddings
                ce_loss.backward(retain_graph=True)
                unnormalized_noise = seq_embed.grad.detach_()
                for p in self.parameters():
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
                norm = unnormalized_noise.norm(p=2, dim=-1)
                normalized_noise = unnormalized_noise / (norm.unsqueeze(dim=-1) + 1e-10)  # add 1e-10 to avoid NaN
                noise_embedding = seq_embed + self.norm_coef * normalized_noise
                _, h_adv = self.rnn(noise_embedding, None)
                h_adv = torch.cat((h_adv[0].squeeze(0), h_adv[1].squeeze(0)), dim=1)
                label_mask = torch.mm(label,label.T).bool().long()
                sup_cont_loss = nt_xent(ht, h_adv, label_mask, cuda=self.use_cuda=='cuda')
                return sup_cont_loss

        elif mode == 'inference':
            _, ht = self.rnn(seq)
            ht = torch.cat((ht[0].squeeze(0), ht[1].squeeze(0)), dim=1)
            logits = self.fc(ht)
            probs = torch.softmax(logits, dim=1)
            return probs, ht

        elif mode == 'validation':
            if self.use_bert:
                seq_embed = self.bert_model(**self.bert_tokenizer(seq, return_tensors='pt', padding=True, truncation=True).to(self.use_cuda))[0]
                seq_embed = self.dropout(seq_embed)
                seq_embed = seq_embed.clone().detach().requires_grad_(True).float()
            else:
                seq_embed = self.embedding(seq)
                seq_embed = self.dropout(seq_embed)
                seq_embed = seq_embed.clone().detach().requires_grad_(True).float()
            _, ht = self.rnn(seq_embed)
            ht = torch.cat((ht[0].squeeze(0), ht[1].squeeze(0)), dim=1)
            logits = self.fc(ht)
            probs = torch.softmax(logits, dim=1)
            return torch.argmax(label, dim=1).tolist(), torch.argmax(probs, dim=1).tolist(), ht

        elif mode == 'test':
            if self.use_bert:
                seq_embed = self.bert_model(**self.bert_tokenizer(seq, return_tensors='pt', padding=True, truncation=True).to(self.use_cuda))[0]
                seq_embed = self.dropout(seq_embed)
                seq_embed = seq_embed.clone().detach().requires_grad_(True).float()
            else:
                seq_embed = self.embedding(seq)
                seq_embed = self.dropout(seq_embed)
                seq_embed = seq_embed.clone().detach().requires_grad_(True).float()
            _, ht = self.rnn(seq_embed)
            ht = torch.cat((ht[0].squeeze(0), ht[1].squeeze(0)), dim=1)
            logits = self.fc(ht)
            probs = torch.softmax(logits, dim=1)
            return probs, ht,logits
        else:
            raise ValueError("undefined mode")











