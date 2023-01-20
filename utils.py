"""
@File   :  utils.py
@Time   :  2022/05/6
@Author  :  Wu Yanan
@Contact :  yanan.wu@bupt.edu.cn
"""
from tkinter.messagebox import NO
from types import new_class
from typing import List
import os
import json
import pandas as pd
import itertools
import matplotlib
from sklearn.metrics import confusion_matrix
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend import set_session
import numpy as np
import random as rn
from sklearn.decomposition import PCA
import torch

# SEED = 123
# tf.random.set_random_seed(SEED)
def setup_seed(SEED):
    np.random.seed(SEED)
    rn.seed(SEED)
    tf.set_random_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)

def naive_arg_topK(matrix, K, axis=1): # 对行元素从小到到排序，返回索引
    full_sort = np.argsort(matrix, axis=axis)
    return full_sort.take(np.arange(K), axis=axis)


def set_allow_growth(device="1"):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.gpu_options.visible_device_list = device
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras


def load_data(dataset):
    texts = []
    labels = []
    partition_to_n_row = {}
    for partition in ['train', 'valid', 'test']:
        with open("./data/" + dataset + "/" + partition + ".seq.in") as fp:
            lines = fp.read().splitlines()
            texts.extend(lines)
            partition_to_n_row[partition] = len(lines)
        with open("./data/" + dataset + "/" + partition + ".label") as fp:
            labels.extend(fp.read().splitlines())

    df = pd.DataFrame([texts, labels]).T
    df.columns = ['text', 'label']
    return df, partition_to_n_row

def load_errordata(errorpath):
    texts = []
    labels = []
    partition_to_n_row = {}
    for partition in ['error']:
        with open(errorpath + "/error.seq.in") as fp:
            lines = fp.read().splitlines()
            texts.extend(lines)
            partition_to_n_row[partition] = len(lines)
        with open(errorpath + "/error.label") as fp:
            labels.extend(fp.read().splitlines())
    df = pd.DataFrame([texts, labels]).T
    df.columns = ['text', 'label']
    return df, partition_to_n_row

def load_errorpreddata(errorpath):
    texts = []
    labels = []
    partition_to_n_row = {}
    for partition in ['error']:
        with open(errorpath + "/error.seq.in") as fp:
            lines = fp.read().splitlines()
            texts.extend(lines)
            partition_to_n_row[partition] = len(lines)
        with open(errorpath + "/error.predlabel") as fp:
            labels.extend(fp.read().splitlines())
    df = pd.DataFrame([texts, labels]).T
    df.columns = ['text', 'label']
    return df, partition_to_n_row

def load_traindata(dataset):
    texts = []
    labels = []
    partition_to_n_row = {}
    for partition in ['train']:
        with open("./data/" + dataset + "/" + partition + ".seq.in") as fp:
            lines = fp.read().splitlines()
            texts.extend(lines)
            partition_to_n_row[partition] = len(lines)
        with open("./data/" + dataset + "/" + partition + ".label") as fp:
            labels.extend(fp.read().splitlines())

    df = pd.DataFrame([texts, labels]).T
    df.columns = ['text', 'label']
    return df, partition_to_n_row

def load_testdata(dataset):
    texts = []
    labels = []
    partition_to_n_row = {}
    for partition in ['test']:
        with open("./data/" + dataset + "/" + partition + ".seq.in") as fp:
            lines = fp.read().splitlines()
            texts.extend(lines)
            partition_to_n_row[partition] = len(lines)
        with open("./data/" + dataset + "/" + partition + ".label") as fp:
            labels.extend(fp.read().splitlines())

    df = pd.DataFrame([texts, labels]).T
    df.columns = ['text', 'label']
    return df, partition_to_n_row

def compute_energy_score(prob,T):
    '''
    Params:
        - logits, (batchsize,cluster_num)
    Returns:
        - energy_score, (batchsize,1)
    '''
    to_np = lambda x: x.data.cpu().numpy()
    prob = torch.from_numpy(prob)
    energy_score = -to_np((T*torch.logsumexp(prob /  T, dim=1)))
    return energy_score

def estimate_best_threshold(seen_m_dist: np.ndarray,
                            unseen_m_dist: np.ndarray) -> float:
    """
    Given mahalanobis distance for seen and unseen instances in valid set, estimate
    a best threshold (i.e. achieving best f1 in valid set) for test set.
    """
    lst = []
    for item in seen_m_dist:
        lst.append((item, "seen"))
    for item in unseen_m_dist:
        lst.append((item, "unseen"))
    # sort by m_dist: [(5.65, 'seen'), (8.33, 'seen'), ..., (854.3, 'unseen')]
    lst = sorted(lst, key=lambda item: item[0])

    threshold = 0.
    tp, fp, fn = len(unseen_m_dist), len(seen_m_dist), 0

    def compute_f1(tp, fp, fn):
        p = tp / (tp + fp + 1e-10)
        r = tp / (tp + fn + 1e-10)
        return (2 * p * r) / (p + r + 1e-10)

    f1 = compute_f1(tp, fp, fn)

    for m_dist, label in lst:   # 
        if label == "seen":  # fp -> tn
            fp -= 1
        else:  # tp -> fn
            tp -= 1
            fn += 1
        if compute_f1(tp, fp, fn) > f1:
            f1 = compute_f1(tp, fp, fn)
            threshold = m_dist + 1e-10

    # print("estimated threshold:", threshold)
    return threshold

def construct_confused_pairs(true_labels,pred_labels,all_labels):
    """ Construct confused label pairs set,
        In order to avoid too many confused pairs, choose at most one per class.
    @ Input:
        
    @ Return:
    """
    error_cm = confusion_matrix(true_labels,pred_labels,all_labels)
    error_pairs = []
    max_error_index = np.argmax(error_cm, axis=1)
    for i in range(error_cm.shape[0]):
        error_pairs.append((all_labels[i],all_labels[max_error_index[i]]))
    return error_pairs

def get_score(cm):
    fs = []
    ps = []
    rs = []
    n_class = cm.shape[0]
    correct = []
    total = []
    for idx in range(n_class):
        TP = cm[idx][idx]
        correct.append(TP)
        total.append(cm[idx].sum())
        r = TP / cm[idx].sum() if cm[idx].sum() != 0 else 0
        p = TP / cm[:, idx].sum() if cm[:, idx].sum() != 0 else 0
        f = 2 * r * p / (r + p) if (r + p) != 0 else 0
        fs.append(f * 100)
        ps.append(p * 100)
        rs.append(r * 100)

    f = np.mean(fs).round(2)
    p = np.mean(ps).round(2)
    r = np.mean(rs).round(2)
    p_seen = np.mean(ps[:-1]).round(2)
    r_seen = np.mean(rs[:-1]).round(2)
    f_seen = np.mean(fs[:-1]).round(2)
    p_unseen = round(ps[-1], 2)
    r_unseen = round(rs[-1], 2)
    f_unseen = round(fs[-1], 2)
    print(f"Overall(macro): , f:{f},  p:{p}, r:{r}")
    print(f"Seen(macro): , f:{f_seen}, p:{p_seen}, r:{r_seen}")
    print(f"=====> Uneen(Experiment) <=====: , f:{f_unseen}, p:{p_unseen}, r:{r_unseen}\n")

    return f, f_seen,  p_seen, r_seen, f_unseen,  p_unseen, r_unseen

def get_errors(cm,classes):
    """  传入混淆矩阵和类别标签，返回topk个错误标签及错误数量 """
    n_class = len(classes)
    errors = {}
    # 计算除了预测为unseen的错误外，其失误数量排序
    for idx in range(n_class):
        tp=cm[idx][idx]
        error = cm[idx].sum()-tp-cm[idx][-1] if cm[idx].sum() != 0 else 0
        errors[classes[idx]]=error
    paixu_group = sorted(errors.items(),key=lambda item:item[1],reverse=True) #[("unseen",74),()]
    top_error_classes = []
    for item in paixu_group:
        top_error_classes.append(item[0]) # ["unseen",""]
    return paixu_group,top_error_classes

def mahalanobis_distance(x: np.ndarray,
                         y: np.ndarray,
                         covariance: np.ndarray) -> float:
    """
    Calculate the mahalanobis distance.

    Params:
        - x: the sample x, shape (num_features,)
        - y: the sample y (or the mean of the distribution), shape (num_features,)
        - covariance: the covariance of the distribution, shape (num_features, num_features)

    Returns:
        - score: the mahalanobis distance in float

    """
    num_features = x.shape[0]

    vec = x - y
    cov_inv = np.linalg.inv(covariance)
    bef_sqrt = np.matmul(np.matmul(vec.reshape(1, num_features), cov_inv), vec.reshape(num_features, 1))
    return np.sqrt(bef_sqrt).item()


def confidence(features: np.ndarray,
               means: np.ndarray,
               distance_type: str,
               cov: np.ndarray = None) -> np.ndarray:
    """
    Calculate mahalanobis or euclidean based confidence score for each class.

    Params:
        - features: shape (num_samples, num_features)
        - means: shape (num_classes, num_features)
        - cov: shape (num_features, num_features) or None (if use euclidean distance)

    Returns:
        - confidence: shape (num_samples, num_classes)
    """
    assert distance_type in ("euclidean", "mahalanobis")

    num_samples = features.shape[0]
    num_features = features.shape[1]
    num_classes = means.shape[0]
    if distance_type == "euclidean":
        cov = np.identity(num_features)

    features = features.reshape(num_samples, 1, num_features).repeat(num_classes,
                                                                     axis=1)  # (num_samples, num_classes, num_features)
    means = means.reshape(1, num_classes, num_features).repeat(num_samples,
                                                               axis=0)  # (num_samples, num_classes, num_features)
    vectors = features - means  # (num_samples, num_classes, num_features)
    cov_inv = np.linalg.inv(cov)
    bef_sqrt = np.matmul(np.matmul(vectors.reshape(num_samples, num_classes, 1, num_features), cov_inv),
                         vectors.reshape(num_samples, num_classes, num_features, 1)).squeeze()
    result = np.sqrt(bef_sqrt)
    result[np.isnan(result)] = 1e12  # solve nan
    return result

def record_global_statistics(test_seen_energy:np.ndarray = None,
                            test_unseen_energy:np.ndarray = None,
                            model_dir:str=None,
                            ori_best_threshold:float=None,
                            ori_better_threshold:float=None):
    ''' 
    Record mathematical statistics of IND and OOD overall energy(max softmax/gda distance) score statistics on the test set.
    Params:

    Return:

    '''
    test_seen_score_mean = np.mean(test_seen_energy)
    test_seen_score_var = np.var(test_seen_energy)
    test_seen_score_max = np.max(test_seen_energy)
    test_seen_score_min = np.min(test_seen_energy)
    test_unseen_score_mean = np.mean(test_unseen_energy)
    test_unseen_score_var = np.var(test_unseen_energy)
    test_unseen_score_max = np.max(test_unseen_energy)
    test_unseen_score_min = np.min(test_unseen_energy)
    original_energyscores = {
        "ModelDir":[model_dir],
        "OriginalBestThreshold":[ori_best_threshold],
        "OriginalBetterThreshold":[ori_better_threshold],
        "TestSeenMean":[test_seen_score_mean],"TestSeenMax":[test_seen_score_max],"TestSeenMin":[test_seen_score_min],"TestSeenVar":[test_seen_score_var],
        "TestUnseenMean":[test_unseen_score_mean],"TestUnseenMax":[test_unseen_score_max],"TestUnseenMin":[test_unseen_score_min],"TestUnseenVar":[test_unseen_score_var]
    }
    return original_energyscores

def record_local_statistics(classes:list=None,
                            ori_best_threshold:float=None,
                            y_test_seen:pd.Series=None,
                            y_pred_unseen:pd.Series=None,
                            test_seen_energy:np.ndarray=None, 
                            test_unseen_energy:np.ndarray=None):
    '''
    Record the mean, variance, max, min of the energy score corresponding to the category.
    Params:
        - classes, The categories in the test set, note that 'ood' is the last in the list.
        - ori_best_threshold,
        - y_test_seen,y_pred_unseen,
        - test_seen_energy, 
        - test_unseen_energy,

    Return:
        - classes_energyscores
    '''
    class_seenground_score_means = {}
    class_seenground_score_vars = {}
    class_seenground_score_mins = {}
    class_seenground_score_maxs = {}
    class_seenground_score_medi = {}
    class_unseenypred_score_means = {}
    class_unseenypred_score_vars = {}
    class_unseenypred_score_mins = {}
    class_unseenypred_score_maxs = {}
    class_unseenypred_score_medi = {}
    class_thresholds = {}
    class_unseenpred_num = {}
    for label in classes[:-1]:
        label_test_indexs = y_test_seen[y_test_seen.isin([label])].index
        label_test_energy = test_seen_energy[label_test_indexs]
        label_ypred_indexs = y_pred_unseen[y_pred_unseen.isin([label])].index
        label_ypred_num = label_ypred_indexs.shape[0] #记录数量
        class_unseenpred_num[label] = label_ypred_num
        label_ypred_energy = test_unseen_energy[label_ypred_indexs]
        threshold = estimate_best_threshold(label_test_energy, label_ypred_energy)
        if threshold == 0:
            threshold = ori_best_threshold
            class_unseenypred_score_means[label] = 0
            class_unseenypred_score_vars[label] = 0
            class_unseenypred_score_mins[label] = 0
            class_unseenypred_score_maxs[label] = 0
            class_unseenypred_score_medi[label] = 0
        else:
            class_unseenypred_score_means[label] = np.mean(label_ypred_energy)
            class_unseenypred_score_vars[label] = np.var(label_ypred_energy)
            class_unseenypred_score_mins[label] = np.min(label_ypred_energy)
            class_unseenypred_score_maxs[label] = np.max(label_ypred_energy)
            class_unseenypred_score_medi[label] = np.median(label_ypred_energy)
        class_seenground_score_means[label] = np.mean(label_test_energy)
        class_seenground_score_vars[label] = np.var(label_test_energy)
        class_seenground_score_mins[label] = np.min(label_test_energy)
        class_seenground_score_maxs[label] = np.max(label_test_energy)
        class_seenground_score_medi[label] = np.median(label_test_energy)
        class_thresholds[label] = threshold

    classes_energyscores = {
        "TestNumber":class_unseenpred_num,
        "TestThreshold":class_thresholds,
        "TestSeenGroundMean":class_seenground_score_means,"TestSeenGroundMedian":class_seenground_score_medi,"TestSeenGroundMax":class_seenground_score_maxs,"TestSeenGroundMin":class_seenground_score_mins,"TestSeenGroundVar":class_seenground_score_vars,
        "TestUnseenPredMean":class_unseenypred_score_means,"TestUnseenGroundMedian":class_unseenypred_score_medi,"TestUnseenPredMax":class_unseenypred_score_maxs,"TestUnseenPredMin":class_unseenypred_score_mins,"TestUnseenPredVar":class_unseenypred_score_vars}
    return classes_energyscores

def record_cm(record_classes,test_data,y_pred,save_path):
    '''Record the confusion matrix for the specified category
    '''
    plot_cm = confusion_matrix(test_data[1], y_pred, labels = record_classes,normalize=None)
    plot_cm_df = pd.DataFrame(plot_cm,columns=record_classes,index = record_classes)
    print(plot_cm_df)
    plot_cm_df.to_csv("./update_result/plot/cm-energygroup-bettershold.csv",mode="a",index=record_classes,header=True)

    top_errors_list,top_errors = get_errors(cm,classes=classes)
    # print(top_errors_list)
    print(top_errors[:15])
    top_errors.remove("unseen")
    draw_classes = top_errors[:14]
    draw_classes.append("unseen")
    topk_cm = confusion_matrix(test_data[1], y_pred, labels = draw_classes)
    print(topk_cm)
    cm_df = pd.DataFrame(cm,columns=classes,index=classes)
    # cm_df = pd.DataFrame(topk_cm,columns=draw_classes,index = draw_classes)
    # print(cm_df)
    cm_df.to_csv(save_path,mode="w",header = True,index=True)

def get_test_info(texts: pd.Series,
                  label: pd.Series,
                  label_mask: pd.Series,
                  softmax_prob: np.ndarray,
                  softmax_classes: List[str],
                  lof_result: np.ndarray = None,
                  gda_result: np.ndarray = None,
                  gda_classes: List[str] = None,
                  save_to_file: bool = False,
                  output_dir: str = None) -> pd.DataFrame:
    """
    Return a pd.DataFrame, including the following information for each test instances:
        - the text of the instance
        - label & masked label of the sentence
        - the softmax probability for each seen classes (sum up to 1)
        - the softmax prediction
        - the softmax confidence (i.e. the max softmax probability among all seen classes)
        - (if use lof) lof prediction result (1 for in-domain and -1 for out-of-domain)
        - (if use gda) gda mahalanobis distance for each seen classes
        - (if use gda) the gda confidence (i.e. the min mahalanobis distance among all seen classes)
    """
    df = pd.DataFrame()
    df['label'] = label
    df['label_mask'] = label_mask
    for idx, _class in enumerate(softmax_classes):
        df[f'softmax_prob_{_class}'] = softmax_prob[:, idx]
    df['softmax_prediction'] = [softmax_classes[idx] for idx in softmax_prob.argmax(axis=-1)]
    df['softmax_confidence'] = softmax_prob.max(axis=-1)
    if lof_result is not None:
        df['lof_prediction'] = lof_result
    if gda_result is not None:
        for idx, _class in enumerate(gda_classes):
            df[f'm_dist_{_class}'] = gda_result[:, idx]
        df['gda_prediction'] = [gda_classes[idx] for idx in gda_result.argmin(axis=-1)]
        df['gda_confidence'] = gda_result.min(axis=-1)
    df['text'] = [text for text in texts]

    if save_to_file:
        df.to_csv(os.path.join(output_dir, "test_info.csv"))

    return df

def log_pred_results(f: float,
                     acc: float,
                     f_seen: float,
                     acc_in: float,
                     p_seen: float,
                     r_seen: float,
                     f_unseen: float,
                     acc_ood: float,
                     p_unseen: float,
                     r_unseen: float,
                     classes: List[str],
                     output_dir: str,
                     confusion_matrix: np.ndarray,
                     ood_loss,
                     adv,
                     cont_loss,
                     threshold: float = None):
    with open(os.path.join(output_dir, "results.txt"), "w") as f_out:
        f_out.write(
            f"Overall:  f1(macro):{f} acc:{acc} \nSeen:  f1(marco):{f_seen} acc:{acc_in} p:{p_seen} r:{r_seen}\n"
            f"=====> Uneen(Experiment) <=====:  f1(marco):{f_unseen} acc:{acc_ood} p:{p_unseen} r:{r_unseen}\n\n"
            f"Classes:\n{classes}\n\n"
            f"Threshold:\n{threshold}\n\n"
            f"Confusion matrix:\n{confusion_matrix}\n"
            f"mode:\nood_loss:{ood_loss}\nadv:{adv}\ncont_loss:{cont_loss}")
    with open(os.path.join(output_dir, "results.json"), "w") as f_out:
        json.dump({
            "f1_overall": f,
            "acc_overall": acc,
            "f1_seen": f_seen,
            "acc_seen": acc_in,
            "p_seen": p_seen,
            "r_seen": r_seen,
            "f1_unseen": f_unseen,
            "acc_unseen": acc_ood,
            "p_unseen": p_unseen,
            "r_unseen": r_unseen,
            "classes": classes,
            "confusion_matrix": confusion_matrix.tolist(),
            "threshold": threshold,
            "ood_loss": ood_loss,
            "adv": adv,
            "cont_loss": cont_loss
        }, fp=f_out, ensure_ascii=False, indent=4)

