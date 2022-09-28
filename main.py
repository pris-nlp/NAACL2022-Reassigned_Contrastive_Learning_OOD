"""
@File   :  train.py
@Time   :  2022/05/6
@Author  :  Wu Yanan
@Contact :  yanan.wu@bupt.edu.cn
"""
import re
import sys
import random
import time
import json
import os
import copy
from utils import *
import argparse
# Parse Arguments
def parse_args():
    parser = argparse.ArgumentParser()
    # arguments need to specify
    parser.add_argument("--mode", type=str, choices=["train","test","both"], default="both", help="Specify running mode: only train, only test or both.")
    parser.add_argument("--detect_method", type=str, default="msp-msplocal-lof-gdamaha-gdamaha_local-gdaeucli-gdaeucli_local-energy-energylocal",help="The settings to detect ood samples, e.g.'energy','energylocal'")
    parser.add_argument("--model_dir", type=str, default=None,help="The directory contains model file (.h5), requried when test only.")
    parser.add_argument("--experiment_No", type=str, default="",help="Manually setting of experiment number.")
    parser.add_argument("--output_dir", type=str, default="./output/",help="The directory to store training models & logs.")
    parser.add_argument("--proportion", type=int, default=None, help="proportion of unseen types to add in, range from 0 to 100.")
    parser.add_argument("--dataset", type=str, default="CLINC_OOD_full",help="The dataset to use, CLINC or SNIPS.")
    parser.add_argument("--unseen_classes", type=str, default=None,help="The specific unseen classes.")
    parser.add_argument("--train_class_num", type=str, choices=["n", "nplus1"], default="n", help="Training classes num, n or n+1")
    parser.add_argument("--use_bert", action="store_true",help="whether to use bert")

    # loss
    parser.add_argument("--scl_cont", action="store_true",
                        help="whether to add supervised contrastive loss")
    parser.add_argument("--rcl_cont", action="store_true",
                        help="whether to add supervised contrastive loss")
    parser.add_argument("--confused_pre_epoches", type=int, default=0,
                        help="Max epoches when in-domain supervised contrastive pre-training (Confused-label pair contrasting).")
    parser.add_argument("--global_pre_epoches", type=int, default=0,
                        help="Max epoches when in-domain supervised contrastive pre-training (Global contrasting).")
    parser.add_argument("--ce_pre_epoches", type=int, default=0,
                        help="Max epoches when in-domain pre-training.")

    # energy
    parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')

    # threshold
    parser.add_argument('--threshold', type=float, default=None, help='detect threshold')

    # default
    parser.add_argument("--unseen_classes_seed", type=int, default=None,
                        help="The random seed to randomly choose seen classes.")
    parser.add_argument("--gpu_device", type=str, default="0",
                        help="The gpu device to use.")
    parser.add_argument("--cuda", action="store_true",
                        help="Whether to use GPU or not.")
    parser.add_argument("--seed", type=int, default=2022,
                        help="Random seed.")
    parser.add_argument("--metrics_path", type=str, default="./result/metrics.csv",
                        help="metrics.csv save path.")

    # model hyperparameters
    parser.add_argument("--embedding_file", type=str,
                        default="./glove_embeddings/glove.6B.300d.txt",
                        help="The embedding file to use.")
    parser.add_argument("--embedding_dim", type=int, default=300,
                        help="The dimension of word embeddings.")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="The dimension of hidden state.")
    parser.add_argument("--contractive_dim", type=int, default=32,
                        help="The dimension of hidden state.")
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument("--norm_coef", type=float, default=0.1,
                    help="coefficients of the normalized adversarial vectors")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="The layers number of lstm.")
    parser.add_argument("--mask_proportion", type=int, default=0,
                        help="proportion of seen class examples to mask, range from 0 to 100.")
    parser.add_argument("--max_seq_len", type=int, default=None,
                        help="The max sequence length. When set to None, it will be implied from data.")
    parser.add_argument("--max_num_words", type=int, default=10000,
                        help="The max number of words.")
    parser.add_argument("--max_epoches", type=int, default=100,
                        help="Max epoches when training.")
    parser.add_argument("--patience", type=int, default=15,
                        help="Patience when applying early stop.")
    parser.add_argument("--lmcl", action="store_true",
                        help="whether to use LMCL loss")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                        help="weight decay")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="learning rate 0.001,2e-5")
    parser.add_argument("--batch_size", type=int, default=200,
                        help="Mini-batch size for train and validation")
    args = parser.parse_args()
    return args

args = parse_args()
setup_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
set_allow_growth(device=args.gpu_device)

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import time
# dataProcess
from keras.utils import to_categorical
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from data_process import DataLoader, CMpairLoader
# Modeling
from model import BiLSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, load_model
from keras import backend as K
from scipy.stats import entropy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Evaluation
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tensorboard_logger import Logger


begin = time.time()
dataset = args.dataset
proportion = args.proportion
NUM_LAYERS = args.num_layers
HIDDEN_DIM = args.hidden_dim
BATCH_SIZE = args.batch_size
EMBEDDING_FILE = args.embedding_file
MAX_SEQ_LEN = args.max_seq_len
MAX_NUM_WORDS = args.max_num_words
EMBEDDING_DIM = args.embedding_dim
CON_DIM = args.contractive_dim
LMCL = args.lmcl
USE_BERT = args.use_bert
SCL_CONT = args.scl_cont
RCL_CONT = args.rcl_cont
USE_CUDA = args.cuda
norm_coef = args.norm_coef
cncnum=0
errornum=0
unseen_classes = args.unseen_classes.split("-")

# Data Process
df, partition_to_n_row = load_data(dataset)
df['content_words'] = df['text'].apply(lambda s: word_tokenize(s))
texts = df['content_words'].apply(lambda l: " ".join(l)) 

# Do not filter out "," and "."
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<UNK>", filters='!"#$%&()*+-/:;<=>@[\]^_`{|}~') 
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts)
sequences_pad = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

# Train-valid-test split
idx_train = (None, partition_to_n_row['train'])
idx_valid = (partition_to_n_row['train'], partition_to_n_row['train'] + partition_to_n_row['valid'])
idx_test = (partition_to_n_row['train'] + partition_to_n_row['valid'], partition_to_n_row['train'] + partition_to_n_row['valid'] + partition_to_n_row['test'])
idx_cont = (partition_to_n_row['train'] + partition_to_n_row['valid'] + partition_to_n_row['test'], None)

X_train = sequences_pad[idx_train[0]:idx_train[1]]
X_valid = sequences_pad[idx_valid[0]:idx_valid[1]]
X_test = sequences_pad[idx_test[0]:idx_test[1]]
X_cont = sequences_pad[idx_cont[0]:idx_cont[1]]

df_train = df[idx_train[0]:idx_train[1]]
df_valid = df[idx_valid[0]:idx_valid[1]]
df_test = df[idx_test[0]:idx_test[1]]
df_cont = df[idx_cont[0]:idx_cont[1]]

y_train = df_train.label.reset_index(drop=True)
y_valid = df_valid.label.reset_index(drop=True)
y_test = df_test.label.reset_index(drop=True)
y_cont = df_cont.label.reset_index(drop=True)
text_train = df_train.text.reset_index(drop=True)
text_valid = df_valid.text.reset_index(drop=True)
text_test = df_test.text.reset_index(drop=True)
text_cont = df_cont.text.reset_index(drop=True)

n_class = y_train.unique().shape[0]
if args.proportion:
    n_class_unseen = round(n_class * proportion/100)

if  args.dataset in ["CLINC_OOD_full_20","CLINC_OOD_full_40","CLINC_OOD_full_60","CLINC_OOD_full_80","CLINC_OOD_full","CLINC_OOD_small","CLINC_OOD_coarse","CLINC_OOD_coarse_100","CLINC_OOD_coarse_real"]:
    y_cols = y_train.unique()
    y_cols_seen = [y_col for y_col in y_cols if y_col not in ["oos"]]
    y_cols_unseen = ["oos"]
elif args.unseen_classes is None:
    if args.unseen_classes_seed is not None:
        random.seed(args.unseen_classes_seed)
        y_cols = y_train.unique()
        y_cols_lst = list(y_cols)
        random.shuffle(y_cols_lst)
        y_cols_unseen = y_cols_lst[:n_class_unseen]
        y_cols_seen = y_cols_lst[n_class_unseen:]
    else:
        # Original implementation
        weighted_random_sampling = False
        if weighted_random_sampling:
            y_cols = y_train.unique()
            y_vc = y_train.value_counts()
            y_vc = y_vc / y_vc.sum()
            y_cols_unseen = np.random.choice(y_vc.index, n_class_unseen, p=y_vc.values, replace=False)
            y_cols_seen = [y_col for y_col in y_cols if y_col not in y_cols_unseen]
        else:
            y_cols_unseen = y_train.value_counts().index[:n_class_unseen]
            y_cols_seen = y_train.value_counts().index[n_class_unseen:]
else:
    y_cols = y_train.unique()
    y_cols_unseen = [y_col for y_col in y_cols if y_col in unseen_classes]
    y_cols_seen = [y_col for y_col in y_cols if y_col not in unseen_classes]

n_class_unseen = len(y_cols_unseen)
if args.unseen_classes is None:
    unseen_classes = str(n_class_unseen)+"_"+"_".join(y_cols_unseen)
else:
    unseen_classes = str(n_class_unseen)+"_"+args.unseen_classes

n_class_seen = len(y_cols_seen)
n_class_unseen = len(y_cols_unseen)
# "...-raw" contains "unseen" and IND data, and the label of OOD in y_train_raw is 'unseen'
# "...-seen" only contains IND data
# "...-unseen" only contains "unseen" data
# Attention: OOD id = the last one; OOD onehot = tensor([0,0,....,0,0,1])
y_train_raw = y_train.copy()
y_valid_raw = y_valid.copy()
y_test_raw = y_test.copy()
X_train_raw = X_train.copy()
X_valid_raw = X_valid.copy()
X_test_raw = X_test.copy()
y_train_raw[y_train_raw.isin(y_cols_unseen)] = "unseen"
y_valid_raw[y_valid_raw.isin(y_cols_unseen)] = 'unseen'
y_test_raw[y_test_raw.isin(y_cols_unseen)] = 'unseen'
y_cols = y_train.unique()

print("*"*100)
print("dataset = ",dataset)
print("unseen classes = ",y_cols_unseen)
print("(Number of dataset) train : valid : test :cont = %d : %d : %d : %d " % (X_train.shape[0], X_valid.shape[0], X_test.shape[0],X_cont.shape[0]))
print(f"(Number of categories) train: valid: test  = {len(list(y_train.unique()))}: {len(list(y_valid.unique()))}: {len(list(y_test.unique()))}")
print("*"*100)

train_seen_idx = y_train[y_train.isin(y_cols_seen)].index
valid_seen_idx = y_valid[y_valid.isin(y_cols_seen)].index
test_seen_idx = y_test[y_test.isin(y_cols_seen)].index
train_unseen_idx = y_train[y_train.isin(y_cols_unseen)].index
valid_unseen_idx = y_valid[y_valid.isin(y_cols_unseen)].index
test_unseen_idx = y_test[y_test.isin(y_cols_unseen)].index

X_train_seen = X_train_raw[train_seen_idx]
y_train_seen = y_train_raw[train_seen_idx]
X_valid_seen = X_valid_raw[valid_seen_idx]
y_valid_seen = y_valid_raw[valid_seen_idx]
X_test_seen = X_test_raw[test_seen_idx]
y_test_seen = y_test_raw[test_seen_idx]
text_train_seen = list(text_train[train_seen_idx])
text_train_unseen = list(text_train[train_unseen_idx])
text_valid_seen = list(text_valid[valid_seen_idx])
text_valid_unseen = list(text_valid[valid_unseen_idx])
text_test_seen = list(text_test[test_seen_idx])
text_test_unseen = list(text_test[test_unseen_idx])
text_train_raw = df_train.text.reset_index(drop=True)
text_valid_raw = df_valid.text.reset_index(drop=True)
text_test_raw = df_test.text.reset_index(drop=True)

X_train_unseen = X_train_raw[train_unseen_idx]
y_train_unseen = y_train_raw[train_unseen_idx]
X_valid_unseen = X_valid_raw[valid_unseen_idx]
y_valid_unseen = y_valid_raw[valid_unseen_idx]
X_test_unseen = X_test_raw[test_unseen_idx]
y_test_unseen = y_test_raw[test_unseen_idx]

le = LabelEncoder()
_classes = list(y_cols_seen)
le.fit(_classes)

y_train_seen_idx = le.transform(y_train_seen)
y_valid_seen_idx = le.transform(y_valid_seen)
y_test_seen_idx = le.transform(y_test_seen)

y_train_seen_onehot = to_categorical(y_train_seen_idx)
y_valid_seen_onehot = to_categorical(y_valid_seen_idx)
y_test_seen_onehot = to_categorical(y_test_seen_idx)

train_data_raw = (X_train_seen, y_train_seen_onehot)
train_data_seen = (X_train_seen,y_train_seen_onehot)
valid_data_raw = (X_valid_seen, y_valid_seen_onehot)
valid_data_seen = (X_valid_seen, y_valid_seen_onehot)
valid_data_unseen = (X_valid_unseen,y_valid_unseen)
test_data_raw = (X_test_seen, y_test_seen_onehot)
test_data_seen = (X_test_seen, y_test_seen_onehot)
test_data_unseen = (X_test_unseen,y_test_unseen)
test_data = (X_test_raw, y_test_raw)


if args.mode in ["train", "both"]:
    train_begin = time.time()
    # GPU setting
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    set_allow_growth(device=args.gpu_device)

    if args.train_class_num == "n":
        X_train_raw = X_train_seen
        y_train_raw = y_train_seen
        y_cols_raw = y_cols_seen
        y_train_raw_idx = y_train_seen_idx
        train_data_raw = train_data_seen
        text_train_raw = text_train_seen
        text_valid_raw = text_valid_seen
        valid_data_raw = valid_data_seen

    timestamp = str(time.time())  # strftime("%m%d%H%M")
    if args.experiment_No:
        output_dir = os.path.join(args.output_dir, f"{dataset}-{unseen_classes}-{args.train_class_num}-{args.seed}-{args.experiment_No}")
    else:
        output_dir = os.path.join(args.output_dir, f"{dataset}-{unseen_classes}-{args.train_class_num}-{args.seed}-{timestamp}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(os.path.join(output_dir, "seen_classes.txt"), "w") as f_out:
        f_out.write("\n".join(y_cols_seen))
    with open(os.path.join(output_dir, "unseen_classes.txt"), "w") as f_out:
        f_out.write("\n".join(y_cols_unseen))

    if not USE_BERT:
        print("Load pre-trained GloVe embedding...")
        MAX_FEATURES = min(MAX_NUM_WORDS, len(word_index)) + 1  # +1 for PAD
        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')
        embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        embedding_matrix = np.random.normal(emb_mean, emb_std, (MAX_FEATURES, EMBEDDING_DIM))
        for word, i in word_index.items():
            if i >= MAX_FEATURES: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix = None

    filepath = os.path.join(output_dir, 'model_best.pkl')
    model = BiLSTM(embedding_matrix, HIDDEN_DIM, NUM_LAYERS, norm_coef,n_class_seen,  LMCL, use_cuda=USE_CUDA, use_bert=USE_BERT, rcl_cont=RCL_CONT, scl_cont=SCL_CONT)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    if args.cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic=True    
        model.cuda()


    best_f1 = 0
    patience = 0
    epoch = 0
    
    if args.rcl_cont:
        # Construct confused label pairs set

        errorpath = re.findall("(.*)-rcl",output_dir)
        print(f"loading errorpath = {errorpath}")
        error_df,_ = load_errordata(errorpath[0])
        error_df, _ = load_errordata(errorpath[0])
        error_df['content_words'] = error_df['text'].apply(lambda s: word_tokenize(s))
        error_texts = error_df['content_words'].apply(lambda l: " ".join(l))
        error_sequences = tokenizer.texts_to_sequences(error_texts)
        MAX_SEQ_LEN = train_data_raw[0].shape[1]
        error_sequences_pad = pad_sequences(error_sequences, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
        X_rcl = error_sequences_pad
        text_rcl_raw = error_df.text.reset_index(drop=True)
        y_rcl = error_df.label.reset_index(drop=True)
        cnc_text = error_df.text.reset_index(drop=True)
        y_rcl_idx = le.transform(y_rcl)
        y_rcl_onehot = to_categorical(y_rcl_idx,np.max(y_train_raw_idx)+1)
        cnc_data_raw = (X_rcl,y_rcl_onehot)
        errornum = X_rcl.shape[0]
        print("errornum = ",errornum)

        errorpred_df,_ = load_errorpreddata(errorpath[0])
        prederrorlabels = errorpred_df["label"].drop_duplicates().values.tolist()
        trueerrorlabels = error_df["label"].drop_duplicates().values.tolist()
        errortypes = list(set(prederrorlabels + trueerrorlabels))
        truetypes = [y_col for y_col in y_cols_seen if y_col not in errortypes]
        y_train_seen_reset = y_train_seen.reset_index(drop=True)

        print(f"confused label set = {len(errortypes)}, clean label set = {len(truetypes)}")
        error_pairs = construct_confused_pairs(error_df["label"],errorpred_df["label"],errortypes)
        print("error_pairs = ",len(error_pairs))

        # Confused-label pair contrasting
        for epoch in range(0,args.confused_pre_epoches):
            global_step = 0
            losses = []
            train_loader = CMpairLoader(y_train_seen_reset,train_data_seen,error_pairs,truetypes,batch_size=200, mode='train', use_bert=USE_BERT, raw_text=text_train_raw)
            train_iterator = tqdm(
                train_loader, initial=global_step,
                desc="Iter (loss=X.XXX)")
            model.train()
            for j, (seq, label) in enumerate(train_iterator):
                
                if args.cuda:
                    if not USE_BERT:
                        seq = seq.cuda()
                    label = label.cuda()
                loss = model(seq, label, mode='ind_pre')
                train_iterator.set_description('Iter (rcl_cont_loss=%5.3f)' % (loss.item()))
                losses.append(loss)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                global_step += 1
            print('RCL Epoch: [{0}] :  Loss {loss:.4f}'.format(epoch, loss=sum(losses)/global_step))
            torch.save(model, filepath)

    if args.scl_cont or args.rcl_cont:
        # Global contrasting
        for epoch in range(0,args.global_pre_epoches):
            global_step = 0
            losses = []
            train_loader = DataLoader(train_data_raw, BATCH_SIZE, use_bert=USE_BERT, raw_text=text_train_raw)
            train_iterator = tqdm(
                train_loader, initial=global_step,
                desc="Iter (loss=X.XXX)")
            model.train()
            for j, (seq, label) in enumerate(train_iterator):
                if args.cuda:
                    if not USE_BERT:
                        seq = seq.cuda()
                    label = label.cuda()
                loss = model(seq, label, mode='ind_pre')
                train_iterator.set_description('Iter (scl_cont_loss=%5.3f)' % (loss.item()))
                losses.append(loss)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                global_step += 1
            print('SCL Epoch: [{0}] :  Loss {loss:.4f}'.format(epoch, loss=sum(losses)/global_step))
            torch.save(model, filepath)

    #  finetune with cross-entropy loss
    for epoch in range(0,args.ce_pre_epoches):
        global_step = 0
        losses = []
        train_loader = DataLoader(train_data_seen, BATCH_SIZE, use_bert=USE_BERT, raw_text=text_train_raw)
        train_iterator = tqdm(train_loader, initial=global_step,desc="Iter (loss=X.XXX)")
        valid_loader = DataLoader(valid_data_seen, BATCH_SIZE, use_bert=USE_BERT, raw_text=text_valid_seen)
        model.train()
        for j, (seq, label) in enumerate(train_iterator): # seq (bactsize,n_class)
            if args.cuda:
                if not USE_BERT:
                    seq = seq.cuda()
                label = label.cuda()
            if epoch == 1:
                loss = model(seq, label, mode='finetune')
            else:
                loss = model(seq, label, mode='finetune')
            train_iterator.set_description('Iter (ce_loss=%5.3f)' % (loss.item()))
            losses.append(loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            global_step += 1
        print('CE Epoch: [{0}] :  Loss {loss:.4f}'.format(epoch, loss=sum(losses)/global_step))

        model.eval()
        predict = []
        target = []
        if args.cuda:
            sim = torch.zeros((n_class_seen, HIDDEN_DIM*2)).cuda()
        else:
            sim = torch.zeros((n_class_seen, HIDDEN_DIM * 2))
        for j, (seq, label) in enumerate(valid_loader):
            if args.cuda:
                if not USE_BERT:
                    seq = seq.cuda()
                label = label.cuda()
            output = model(seq, label, mode='validation')
            target += output[0]
            predict += output[1]
            sim += torch.mm(label.T, output[2])
        sim = sim / len(predict)
        n_sim = sim.norm(p=2, dim=1, keepdim=True)
        sim = (sim @ sim.t()) / (n_sim * n_sim.t()).clamp(min=1e-8)
        if args.cuda:
            sim = sim - 1e4 * torch.eye(n_class_seen).cuda()
        else:
            sim = sim - 1e4 * torch.eye(n_class_seen)
        sim = torch.softmax(sim, dim=1)
        f1 = metrics.f1_score(target, predict, average='macro')
        if f1 > best_f1:
            torch.save(model, filepath)
            best_f1 = f1
        print('f1:{f1:.4f}'.format(f1=f1))

if args.mode in ["test", "both"]:
    test_begin = time.time()
    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = os.path.join(args.output_dir, f"{dataset}-{unseen_classes}-{args.train_class_num}-{args.seed}-{args.experiment_No}")
    
    model = torch.load(os.path.join(model_dir, "model_best.pkl"), map_location='cuda:'+args.gpu_device)

    if args.train_class_num == "n":
        X_train_raw = X_train_seen
        y_cols_raw = y_cols_seen
        y_train_raw_idx = y_train_seen_idx
        train_data_raw = train_data_seen
        text_train_raw = text_train_seen

    train_loader = DataLoader(train_data_raw, BATCH_SIZE, mode="test",use_bert=USE_BERT, raw_text=text_train_raw)
    valid_loader = DataLoader(valid_data_raw, BATCH_SIZE, mode="validation",use_bert=USE_BERT, raw_text=text_valid_raw)
    test_loader = DataLoader(test_data, BATCH_SIZE, mode="test",use_bert=USE_BERT, raw_text=text_test_raw)
    train_loader_seen = DataLoader(train_data_seen, BATCH_SIZE,mode="test", use_bert=USE_BERT, raw_text=text_train_seen)
    vali_train_loader_seen = DataLoader(train_data_seen, BATCH_SIZE,use_bert=USE_BERT, raw_text=text_train_seen)
    valid_loader_seen = DataLoader(valid_data_seen, BATCH_SIZE,mode="test", use_bert=USE_BERT, raw_text=text_valid_seen)
    test_loader_seen = DataLoader(test_data_seen, BATCH_SIZE, mode="test",use_bert=USE_BERT, raw_text=text_test_seen)
    valid_loader_unseen = DataLoader(valid_data_unseen, BATCH_SIZE, mode="test",use_bert=USE_BERT, raw_text=text_valid_unseen)
    test_loader_unseen = DataLoader(test_data_unseen, BATCH_SIZE, mode="test",use_bert=USE_BERT, raw_text=text_test_unseen)

    torch.no_grad()
    model.eval()
    predict = []
    target = []

    classes = list(le.classes_)+['unseen']
    feature_train = None
    feature_train_seen = None
    feature_valid_unseen = None
    feature_valid_seen = None
    feature_test = None
    prob_train = None
    prob_train_seen = None
    prob_valid_unseen = None
    prob_valid_seen = None
    prob_test_unseen = None
    prob_test_seen = None
    prob_test = None
    logit_valid_unseen = None
    logit_valid_seen = None
    logit_test = None

    for j, (seq, label) in enumerate(train_loader):
        if args.cuda:
            if not USE_BERT: seq = seq.cuda()
        output = model(seq,  mode = 'test')
        if j == 0:
            feature_train = output[1]
            prob_train = output[0]
        else:
            feature_train = torch.cat((feature_train,output[1]),dim=0)
            prob_train = torch.cat((prob_train, output[0]), dim=0)


    for j, (seq, label) in enumerate(train_loader_seen):
        if args.cuda:
            if not USE_BERT: seq = seq.cuda()
        output = model(seq,  mode = 'test')
        if j == 0:
            feature_train_seen = output[1]
            prob_train_seen = output[0]
        else:
            feature_train_seen = torch.cat((feature_train_seen,output[1]),dim=0)
            prob_train_seen = torch.cat((prob_train_seen, output[0]), dim=0)


    for j, (seq, label) in enumerate(valid_loader_unseen):
        if args.cuda:
            if not USE_BERT: seq = seq.cuda()
        output = model(seq,  None, mode = 'test')
        if j > 0:
            feature_valid_unseen = torch.cat((feature_valid_unseen,output[1]),dim=0)
            prob_valid_unseen = torch.cat((prob_valid_unseen,output[0]),dim=0)
            logit_valid_unseen = torch.cat((logit_valid_unseen,output[2]),dim=0)
        else:
            feature_valid_unseen = output[1]
            prob_valid_unseen = output[0]
            logit_valid_unseen = output[2]

    for j, (seq, label) in enumerate(valid_loader_seen):
        if args.cuda:
            if not USE_BERT:seq = seq.cuda()
        output = model(seq,  None, mode = 'test')
        if j > 0:
            feature_valid_seen = torch.cat((feature_valid_seen,output[1]),dim=0)
            prob_valid_seen = torch.cat((prob_valid_seen,output[0]),dim=0)
            logit_valid_seen = torch.cat((logit_valid_seen,output[2]),dim=0)
        else:
            feature_valid_seen = output[1]
            prob_valid_seen = output[0]
            logit_valid_seen = output[2]

    for j, (seq, label) in enumerate(test_loader_unseen):
        if args.cuda:
            if not USE_BERT:seq = seq.cuda()
        output = model(seq, None, mode = 'test')
        if j > 0:
            feature_test_unseen = torch.cat((feature_test_unseen,output[1]),dim=0)
            prob_test_unseen = torch.cat((prob_test_unseen,output[0]),dim=0)
            logit_test_unseen = torch.cat((logit_test_unseen,output[2]),dim=0)
        else:
            feature_test_unseen = output[1]
            prob_test_unseen = output[0]
            logit_test_unseen = output[2]

    for j, (seq, label) in enumerate(test_loader_seen):
        if args.cuda:
            if not USE_BERT:seq = seq.cuda()
        output = model(seq, None, mode = 'test')
        if j > 0:
            feature_test_seen = torch.cat((feature_test_seen,output[1]),dim=0)
            prob_test_seen = torch.cat((prob_test_seen,output[0]),dim=0)
            logit_test_seen = torch.cat((logit_test_seen,output[2]),dim=0)
        else:
            feature_test_seen = output[1]
            prob_test_seen = output[0]
            logit_test_seen = output[2]

    for j, (seq, label) in enumerate(test_loader):
        if args.cuda:
            if not USE_BERT:seq = seq.cuda()
        output = model(seq, None, mode = 'test')
        if j > 0:
            feature_test = torch.cat((feature_test,output[1]),dim=0)
            prob_test = torch.cat((prob_test, output[0]), dim=0)
            logit_test = torch.cat((logit_test, output[2]), dim=0)
        else:
            feature_test = output[1]
            prob_test = output[0]
            logit_test = output[2]

    feature_train = feature_train.cpu().detach().numpy()
    feature_train_seen = feature_train_seen.cpu().detach().numpy()
    feature_valid_unseen = feature_valid_unseen.cpu().detach().numpy()
    feature_valid_seen = feature_valid_seen.cpu().detach().numpy()
    feature_test = feature_test.cpu().detach().numpy()
    feature_test_unseen = feature_test_unseen.cpu().detach().numpy()
    feature_test_seen = feature_test_seen.cpu().detach().numpy()
    prob_train = prob_train.cpu().detach().numpy()
    prob_train_seen = prob_train_seen.cpu().detach().numpy()
    prob_valid_seen = prob_valid_seen.cpu().detach().numpy()
    prob_valid_unseen = prob_valid_unseen.cpu().detach().numpy()
    prob_test = prob_test.cpu().detach().numpy()
    prob_test_seen = prob_test_seen.cpu().detach().numpy()
    prob_test_unseen = prob_test_unseen.cpu().detach().numpy()
    logit_valid_seen = logit_valid_seen.cpu().detach().numpy()
    logit_valid_unseen = logit_valid_unseen.cpu().detach().numpy()
    logit_test = logit_test.cpu().detach().numpy()
    logit_test_seen = logit_test_seen.cpu().detach().numpy()
    logit_test_unseen = logit_test_unseen.cpu().detach().numpy()


    # Record the error cases in the training set.
    errorsave_path = model_dir
    train_df,_ = load_traindata(dataset)
    train_text_pd = train_df["text"][train_seen_idx].reset_index(drop=True)
    train_label_pd = train_df["label"][train_seen_idx].reset_index(drop=True)
    if args.detect_method == "nplus1":
        train_classes = le.classes_
        df_train = pd.DataFrame(prob_train, columns=le.classes_)
    else:
        train_classes = le.classes_
        df_train = pd.DataFrame(prob_train, columns=le.classes_)
    y_pred_train = df_train.idxmax(axis=1)
    y_pred_train_idx = np.argmax(prob_train,axis = 1)
    y_true_train_idx = y_train_raw_idx
    error_idx = np.flatnonzero(y_pred_train_idx - y_true_train_idx)
    error_seq = train_text_pd[error_idx]
    error_lab = train_label_pd[error_idx]
    error_predlab = y_pred_train[error_idx]
    errornum = error_idx.shape[0]
    error_seq.to_csv(os.path.join(errorsave_path,"error.seq.in"),index=0,header=0)
    error_lab.to_csv(os.path.join(errorsave_path,"error.label"),index=0,header=0)
    error_predlab.to_csv(os.path.join(errorsave_path,"error.predlabel"),index=0,header=0)
    print("*"*100)
    print("train error  case number = ",errornum)


    # Record the error cases in the Test set.
    errorsave_path = model_dir
    test_df,_ = load_testdata(dataset)
    test_text_pd = test_df["text"][test_seen_idx].reset_index(drop=True)
    test_label_pd = test_df["label"][test_seen_idx].reset_index(drop=True)
    if args.detect_method == "nplus1":
        test_classes = le.classes_
        df_test = pd.DataFrame(prob_test_seen, columns=le.classes_)
    else:
        test_classes = le.classes_
        df_test = pd.DataFrame(prob_test_seen, columns=le.classes_)
    y_pred_test = df_test.idxmax(axis=1)
    y_pred_test_idx = np.argmax(prob_test_seen,axis = 1)
    y_true_test_idx = y_test_seen_idx
    error_idx = np.flatnonzero(y_pred_test_idx - y_true_test_idx)
    error_seq = test_text_pd[error_idx]
    error_lab = test_label_pd[error_idx]
    error_predlab = y_pred_test[error_idx]
    errornum = error_idx.shape[0]
    error_seq.to_csv(os.path.join(errorsave_path,"testseen_error.seq.in"),index=0,header=0)
    error_lab.to_csv(os.path.join(errorsave_path,"testseen_error.label"),index=0,header=0)
    error_predlab.to_csv(os.path.join(errorsave_path,"testseen_error.predlabel"),index=0,header=0)
    print("test seen error case number = ",errornum)

    detect_method_list = args.detect_method.split("-")
    for detect_method in detect_method_list:
        save_dir = os.path.join(model_dir,detect_method)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        classscore_savename = re.findall(args.dataset+"(.*)",model_dir)[0] + ".csv"
        
        if detect_method == "energylocal":
            print("********************************* detect_method = energy (local) *********************************")
            method = "energylocal"
            # compute energy score
            valid_seen_energy = compute_energy_score(logit_valid_seen,args.T)
            valid_unseen_energy = compute_energy_score(logit_valid_unseen,args.T)
            test_seen_energy = compute_energy_score(logit_test_seen,args.T)
            test_unseen_energy = compute_energy_score(logit_test_unseen,args.T)

            # Automatic thresholding using validation set
            if args.threshold:
                ori_better_threshold = args.threshold
            else:
                ori_better_threshold = estimate_best_threshold(valid_seen_energy, valid_unseen_energy)
            
            # predict
            y_valid_seen = y_valid_seen.reset_index(drop=True)
            y_test_seen = y_test_seen.reset_index(drop=True)
            df_all = pd.DataFrame(prob_test, columns=le.classes_)
            df_all['unseen'] = 0
            y_pred = df_all.idxmax(axis=1)
            y_pred_score = pd.Series(compute_energy_score(logit_test,args.T))

            df_unseen = pd.DataFrame(prob_test_unseen, columns=le.classes_)
            df_unseen['unseen'] = 0
            y_pred_unseen = df_unseen.idxmax(axis=1)

            y_pred_unseen_score = pd.Series(compute_energy_score(logit_test_unseen,args.T))
            df_valid_unseen = pd.DataFrame(prob_valid_unseen, columns=le.classes_)
            df_valid_unseen['unseen'] = 0
            y_predvalid_unseen = df_valid_unseen.idxmax(axis=1)
            y_predvalid_unseen_score = pd.Series(compute_energy_score(logit_valid_unseen,args.T))

            # Adaptive Thresholding
            # Find the most suitable threshold value on the validation set: 
            # compare "Energy{validation set GroudTruth is the class's seen data}" and "Energy{validation set is predicted to be the unseen data for the class}" to get the optimal threshold.
            # If there is no predicted sample of this class, set the energy threshold of this class as the original threshold.
            y_pred_thresholds = copy.deepcopy(y_pred)
            thresholds = {}
            for label in classes[:-1]:
                label_valid_indexs = y_valid_seen[y_valid_seen.isin([label])].index 
                label_valid_energy = valid_seen_energy[label_valid_indexs]
                label_ypred_indexs = y_predvalid_unseen[y_predvalid_unseen.isin([label])].index
                label_ypred_energy = valid_unseen_energy[label_ypred_indexs]
                threshold = estimate_best_threshold(label_valid_energy, label_ypred_energy)
                if threshold == 0:
                    threshold = ori_better_threshold
                thresholds[label] = [threshold]
                label_pred_indexs = y_pred[y_pred.isin([label])].index
                y_pred_thresholds[label_pred_indexs] = threshold
            y_pred_score_threshold = y_pred_score - y_pred_thresholds
            y_pred[y_pred_score_threshold[y_pred_score_threshold > 0].index] = 'unseen'

            # Calculate F1, p, r
            cm = confusion_matrix(test_data[1], y_pred, labels = classes)
            f, f_seen, p_seen, r_seen, f_unseen, p_unseen, r_unseen = get_score(cm)
            metrics_name = ["dataset","unseen_classes","train_class_num","detect_method","seed","T", "f1_ood","r_ood","p_ood","f1_ind","r_ind","p_ind", "f1","model_dir","confused_epoches","global_epoches","ce_epoches"]
            metrics_data = np.array([[args.dataset,unseen_classes,args.train_class_num,detect_method,args.seed,args.T, f_unseen,r_unseen,p_unseen, f_seen,r_seen,p_seen, f,model_dir,args.confused_pre_epoches,args.global_pre_epoches,args.ce_pre_epoches]])
            metrics_df = pd.DataFrame(metrics_data,columns=metrics_name)
            metrics_df.to_csv(args.metrics_path,mode="a",header = False,index=None)