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

def construct_confused_pairs(true_labels,pred_labels,all_labels):
    """ Construct confused label pairs set,
        In order to avoid too many confused pairs, choose at most one per class.
    @ Input:
        
    @ Return:
    """
    error_cm = confusion_matrix(true_labels,pred_labels,all_labels)
    # x_idx,y_idx = np.nonzero(error_cm)
    # error_pairs = []
    # for i in range(x_idx.shape[0]):
    #     error_pairs.append((all_labels[x_idx[i]],all_labels[y_idx[i]]))
    # print("error_pairs = ",len(error_pairs),error_pairs)
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

def plot_confusion_matrix(output_dir, cm, classes, normalize=False,
                          title='Confusion matrix', figsize=(12, 10),
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # Compute confusion matrix
    np.set_printoptions(precision=2)
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mat.png"))


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

def com_IntraVar_InterDistance(feature_train_seen,y_train_seen,prob_train_seen,feature_test_seen,prob_test_seen,feature_test,test_data):
    # Calculate intra-class variance and inter-class distance
    '''
    @ Input:

    @ Return:
    '''
    print("********************************* 计算类内方差 类间距离 *********************************")
    var_and_mean_dis(feature_train_seen, labels=y_train_seen,norm = True)
    var_and_mean_dis(feature_train_seen, labels=y_train_seen,norm = False)
    pairs = [("change_user_name","user_name"),("change_user_name","change_ai_name"),("user_name","what_is_your_name"),("change_ai_name","what_is_your_name"),("redeem_rewards","rewards_balance"),("ingredients_list","recipe"),("shopping_list","shopping_list_update"),("play_music","change_speed"),("payday","change_user_name"),("distance","directions")]
    for pair in pairs:
        print("\n")
        cm_pair = list(pair)
        cm_pair_index = y_train_seen[y_train_seen.isin(cm_pair)].index
        feature_train_pairs = feature_train_seen[cm_pair_index]
        y_train_pairs = list(y_train_seen[cm_pair_index])

        inter_dis(feature_train_pairs, labels=y_train_pairs,norm = True)
        inter_dis(feature_train_pairs, labels=y_train_pairs,norm = False)
        intra_var(feature_train_pairs, labels=y_train_pairs,norm = True)
        intra_var(feature_train_pairs, labels=y_train_pairs,norm = False)

    print("########### probs train  ########")
    var_and_mean_dis(prob_train_seen, labels=y_train_seen)
    y_test_seen = y_test_seen.tolist()
    print("########### feature test ########")
    var_and_mean_dis(feature_test_seen, labels=y_test_seen)
    print("########### probs test  ########")
    var_and_mean_dis(prob_test_seen, labels=y_test_seen)

    print("*********************************  pca *********************************")
    pca_labels = ["change_user_name","user_name","redeem_rewards","rewards_balance","ingredients_list","recipe","shopping_list","shopping_list_update","change_ai_name","what_is_your_name","play_music","change_speed"]
    pca_index = test_data[1][test_data[1].isin(pca_labels)].index
    print(feature_test[pca_index].shape)

    ids = le.transform(pca_labels)
    id2label = {}
    for i in range(len(pca_labels)):
        id2label[ids[i]]=pca_labels[i]
    print("id2label = ",id2label)
    pca_visualization(feature_test[pca_index],pd.Series(test_data[1])[pca_index],classes = pca_labels,save_path = "result/plot/pca2.png")
    plot_t_sne(feature_test[pca_index],pd.Series(test_data[1])[pca_index], sample_num = 30, id2label = id2label,save_path = "result/plot/pca0.png")

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
    plot_cm_df.to_csv("./update_result/plot/1114-cm-energygroup-bettershold.csv",mode="a",index=record_classes,header=True)

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
    tp, fp, fn = len(unseen_m_dist), len(seen_m_dist), 0 # 起始阈值在最左边，所有数据都预测为unseen，（unseen视为正类positive）

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


def pca_visualization(X: np.ndarray,
                      y: pd.Series,
                      classes: List[str],
                      save_path: str):
    """
    Apply PCA visualization for features.
    """
    print("#### mode = pca ####")
    red_features = PCA(n_components=2, svd_solver="full").fit_transform(X)

    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots()
    for _class in classes:
        if _class == "unseen":
            ax.scatter(red_features[y == _class, 0], red_features[y == _class, 1],
                       label=_class, alpha=0.5, s=20, edgecolors='none', color="gray")
        else:
            ax.scatter(red_features[y == _class, 0], red_features[y == _class, 1],
                       label=_class, alpha=0.5, s=20, edgecolors='none', zorder=10)
    ax.legend()
    ax.grid(True)
    plt.savefig(save_path, format="png")


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

#!/usr/bin/env python


def plot_t_sne(embeddings, labels, sample_num, id2label,save_path):
    """
    embeddings: 样本表示tensor
    labels: 标签list
    sample_num: 采样个数
    id2label: id到标签字典
    """
    print("##### mode 1")
    embeddings = embeddings.tolist()
    # 采样200个样本
    reps_dict = {}
    num_dict = {}
    sample_embeddings = []
    sample_labels = []
    for l in set(labels):
        num_dict[l] = 0

    for i in range(len(labels)):
        if num_dict[labels[i]] <= sample_num:
            reps_dict.setdefault(labels[i], []).append(embeddings[i])
            sample_embeddings.append(embeddings[i])
            sample_labels.append(labels[i])
            num_dict[labels[i]] = num_dict[labels[i]] + 1

    # t-sne
    from sklearn.manifold import TSNE
    iters = 50000
    tsne_sample = TSNE(perplexity=30, n_components=2, init='pca', n_iter=iters, learning_rate= 0.5) # TSNE降维，降到2
    relation_expresentation_sample = tsne_sample.fit_transform(sample_embeddings)
    x_min, x_max = relation_expresentation_sample.min(0), relation_expresentation_sample.max(0)
    X_norm = (relation_expresentation_sample - x_min) / (x_max - x_min)  # 归一化
    # plot
    plt.figure(figsize=(4, 4))

#     id2label = {2:'instance of', 3:'in country', 6:'capital of', 7:'director of', 8:'has child'}
    reps_dict = {}
    for i in range(len(sample_labels)):
        reps_dict.setdefault(sample_labels[i], []).append((relation_expresentation_sample[i, 0], relation_expresentation_sample[i, 1]))

    ids = list(id2label.keys())
    for key in reps_dict:
        reps = []
        if key in ids:  # 挑选所有类中的要画的类list
            if key == 18:
                color = '#82cbb2'
                for i in reps_dict[key]:
                    x = i[0]
                    y = i[1]
                    if x<5 and y>0:
                        reps.append(i)
            elif key == 31:
                color = '#db5856'
                for i in reps_dict[key]:
                    x = i[0]
                    y = i[1]
                    if x>0 and y>40:
                        reps.append(i)
            elif key == 130:
                color = '#fdaa48'
                for i in reps_dict[key]:
                    x = i[0]
                    y = i[1]
                    if x>0 and y<20 and y>-30:
                        reps.append(i)
            elif key == 36:
                color = '#25a36f'
                reps = reps_dict[key][:200] # 画前两百个

            elif key == 102:
                color = '#0e87bc'
                reps = reps_dict[key][:200]
            # elif key == 102:
            #     color = '#0e87bc'
            #     reps = reps_dict[key][:200]
            # elif key == 102:
            #     color = '#0e87bc'
            #     reps = reps_dict[key][:200]
#             reps = reps_dict[key][:200]
            print(len(reps))
            x, y = zip(*reps)
    #         plt.plot(x, y, 'o', alpha=1, label='{}'.format(key)) 
            plt.scatter(x, y, alpha=1, label='{}'.format(id2label[key]), linewidths=1, edgecolors='w', color=color, s=50)
    plt.xticks()
    plt.yticks()
    plt.legend(loc='upper right', handlelength=0.8)
    # plt.grid(True, linestyle='--')  # linestyle样式 linewidth 网格宽度 color颜色
    plt.title('(a) mode0', y=-0.18, fontsize=10) # 标题
    plt.savefig(save_path, format='png', bbox_inches = 'tight')  # 保存为pdf
    # plt.show()
    

    
import torch.nn.functional as F
import numpy as np
def intra_var(embeddings, labels,norm=False):
    """
    计算两个类的方差
    - embeddings: 样本表示list or numpy
    - labels: 标签
    - norm: 是否归一化
    """
    # 计算类内方差，分析所有类方差的最大值，最小值，均值，中值
    # 得到每个类样本表示
    reps_class_dict = {}
    var_class_dict = {}
    val_class_list = []
    torch.set_printoptions(precision=16)
    for i in range(len(labels)):
        if norm:
            reps_class_dict.setdefault(labels[i], []).append(F.normalize(torch.tensor(embeddings[i]), dim=0).numpy())
        else:
            reps_class_dict.setdefault(labels[i], []).append(embeddings[i])

    for key in reps_class_dict:
        reps_class = reps_class_dict[key]
        reps_class = torch.tensor(reps_class)
        # var_class_dict[key] = torch.var(reps_class)  #  元素方差
        # val_class_list.append(torch.var(reps_class))
        # 向量方差
        class_mean = torch.mean(reps_class, dim=0)
        tss = 0
        for i in range(len(reps_class)):
            tss += torch.sum((reps_class[i]-class_mean)**2)
        # print(tss/len(reps_class))
        var_class_dict[key] = tss/len(reps_class)
        val_class_list.append(tss/len(reps_class))
        print("\"{0}\" 的类内方差为：{1}".format(key,var_class_dict[key]))
        # 向量方差
    val_class_list = torch.tensor(val_class_list)
    

    # print('————————————————')
    # print('计算所有的类内方差')
    # print('最小值:', torch.min(val_class_list))
    # print('最大值:', torch.max(val_class_list))
    # print('中位数:', torch.median(val_class_list))
    # print('均值:', torch.mean(val_class_list))

def inter_dis(embeddings, labels,norm=False):
    # 计算两类之间的距离
    reps_class_dict = {}
    torch.set_printoptions(precision=16)
    for i in range(len(labels)):
        if norm:
            reps_class_dict.setdefault(labels[i], []).append(F.normalize(torch.tensor(embeddings[i]), dim=0).numpy())
        else:
            reps_class_dict.setdefault(labels[i], []).append(embeddings[i])

    k = [1]
    classes = list(set(labels))
    reps_class_center = {}
    for key in reps_class_dict:
        reps_class_tensor = torch.tensor(reps_class_dict[key])
        reps_class_center[key] = torch.mean(reps_class_tensor, dim=0)
    distance_class_dict = {}
    for key in reps_class_center:
        vector1 = reps_class_center[key].unsqueeze(0)
        for key1 in reps_class_center:
            if key1 != key:
                vector2 = reps_class_center[key1].unsqueeze(0)
                dis = F.pairwise_distance(vector1, vector2, p=2)
                distance_class_dict.setdefault(key, []).append(dis.numpy())
    for key in distance_class_dict:
        distance_class_dict[key].sort()

    mean_dis_k = {}
    for i in k:
        all_mean_dis = 0
        for key in distance_class_dict:
            mean_dis = torch.mean(torch.tensor(distance_class_dict[key][:i]))
            all_mean_dis += mean_dis
        all_mean_dis = all_mean_dis/len(set(labels))
        mean_dis_k[i] = all_mean_dis

    for key in mean_dis_k:
        print('\"{0}\" 与 \"{1}\" 之间的距离为: {2}'.format(classes[0],classes[1], mean_dis_k[key]))

def var_and_mean_dis(embeddings, labels,norm):
    """
    embeddings: 样本表示list or numpy
    labels: 标签
    """
    # 计算类内方差，分析所有类方差的最大值，最小值，均值，中值
    # 得到每个类样本表示
    reps_class_dict = {}
    var_class_dict = {}
    val_class_list = []
    torch.set_printoptions(precision=16)
    for i in range(len(labels)):
        if norm:
            reps_class_dict.setdefault(labels[i], []).append(F.normalize(torch.tensor(embeddings[i]), dim=0).numpy())
        else:
            reps_class_dict.setdefault(labels[i], []).append(embeddings[i])

    for key in reps_class_dict:
        reps_class = reps_class_dict[key]
        reps_class = torch.tensor(reps_class)
        # var_class_dict[key] = torch.var(reps_class)  #  元素方差
        # val_class_list.append(torch.var(reps_class))
        # 向量方差
        class_mean = torch.mean(reps_class, dim=0)
        tss = 0
        for i in range(len(reps_class)):
            tss += torch.sum((reps_class[i]-class_mean)**2)
        # print(tss/len(reps_class))
        var_class_dict[key] = tss/len(reps_class)
        val_class_list.append(tss/len(reps_class))
        # 向量方差
    val_class_list = torch.tensor(val_class_list)


    print('————————————————')
    print('计算所有的类内方差')
    print('最小值:', torch.min(val_class_list))
    print('最大值:', torch.max(val_class_list))
    print('中位数:', torch.median(val_class_list))
    print('均值:', torch.mean(val_class_list))
    print('————————————————')


    # 计算类间距离，分析每个类到最近1，2，4，6，8，9个类的平均类间距

    k = [1, 5, 10, 30, 50]
    reps_class_center = {}
    for key in reps_class_dict:
        reps_class_tensor = torch.tensor(reps_class_dict[key])
        reps_class_center[key] = torch.mean(reps_class_tensor, dim=0)
    distance_class_dict = {}
    for key in reps_class_center:
        vector1 = reps_class_center[key].unsqueeze(0)
        for key1 in reps_class_center:
            if key1 != key:
                vector2 = reps_class_center[key1].unsqueeze(0)
                dis = F.pairwise_distance(vector1, vector2, p=2)
                distance_class_dict.setdefault(key, []).append(dis.numpy())
    for key in distance_class_dict:
        distance_class_dict[key].sort()

    mean_dis_k = {}
    for i in k:
        all_mean_dis = 0
        for key in distance_class_dict:
            mean_dis = torch.mean(torch.tensor(distance_class_dict[key][:i]))
            all_mean_dis += mean_dis
        all_mean_dis = all_mean_dis/len(set(labels))
        mean_dis_k[i] = all_mean_dis

    print('————————————————')
    for key in mean_dis_k:
        print('与最近{0}个类的平均类间距离: {1}'.format(key, mean_dis_k[key]))
    print('————————————————')

def single_class_pca(X: np.ndarray,
                     y: pd.Series,
                     gda_means: np.ndarray,
                     gda_cov: np.ndarray,
                     gda_classes: List[str],
                     save_dir: str):
    """
    Compare Euclidean and Mahalanobis distance using PCA visualization.

    For the data transformation, see: https://zhuanlan.zhihu.com/p/45140262
    """
    def center(data, mean):
        return data - mean
    def whiten(data, cov):
        eig_vals, eig_vecs = np.linalg.eig(cov)
        decorrelated = data.dot(eig_vecs)  # Apply the eigenvectors to X
        whitened = decorrelated / np.sqrt(eig_vals + 1e-9)  # Rescale the decorrelated data
        return whitened
    def get_2d_data(X, y, _class, gda_classes, gda_means, gda_cov):
        mean = gda_means[list(gda_classes).index(_class)]
        selected_data = X[y == _class]
        centered_data = center(selected_data, mean)
        transformed_data = whiten(centered_data, gda_cov)
        bef = PCA(n_components=2, svd_solver="full").fit_transform(centered_data)
        aft = PCA(n_components=2, svd_solver="full").fit_transform(transformed_data)
        return bef, aft
    plt.style.use("seaborn-darkgrid")
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    fig = plt.figure(figsize=(7.2, 3.6))
    ax1 = fig.add_axes([0.05, 0.06, 0.43, 0.86])
    ax2 = fig.add_axes([0.53, 0.06, 0.43, 0.86])
    # centers_1 = {
    #     "AddToPlaylist": (0.0, 0.0),
    #     "BookRestaurant": (-20.0, 0.0),
    #     "PlayMusic": (10.0, 15.0),
    #     "RateBook": (10.0, -13.0)
    # }
    # centers_2 = {
    #     "AddToPlaylist": (0.0, 0.0),
    #     "BookRestaurant": (-30.0, 0.0),
    #     "PlayMusic": (15.0, 22.5),
    #     "RateBook": (15.0, -19.5)
    # }
    centers_1 = {
        "AddToPlaylist": (2.0, -2.0),
        "BookRestaurant": (-20.0, 0.0),
        "PlayMusic": (16.0, 10.0),
        "RateBook": (12.0, -18.0)
    }
    centers_2 = {
        "AddToPlaylist": (3.0, 1.0),
        "BookRestaurant": (-17.0, 0.0),
        "PlayMusic": (20.0, 29.5),
        "RateBook": (15.0, -17.5)
    }
    param_1 = {
        "AddToPlaylist": ((-12.0, -10.0), 2.0),
        "BookRestaurant": ((10.0, 5.0), 2.0),
        "PlayMusic": ((0.0, 0.0), 3.5),
        "RateBook": ((0.0, 12.0), 3.5)
    }
    param_2 = {
        "AddToPlaylist": ((-15.0, 10.0), 2.0),
        "BookRestaurant": ((10.0, -9.0), 2.5),
        "PlayMusic": ((-5.0, 1.0), 2.0),
        "RateBook": ((-1.0, 2.0), 3.0)
    }
    new_centers_1 = {
        "AddToPlaylist": (1, -2),
        "BookRestaurant": (-20.5, 0.0),
        "PlayMusic": (15, 6.0),
        "RateBook": (11.5, -18)
    }
    new_centers_2 = {
        "AddToPlaylist": (0.0, 0.0),
        "BookRestaurant": (-18.0, -1.0),
        "PlayMusic": (13, 15),
        "RateBook": (12.5, -18)
    }
    for _class in gda_classes:
        center_x1, center_y1 = centers_1[_class]
        center_x2, center_y2 = centers_2[_class]
        data_1, data_2 = get_2d_data(X, y, _class, gda_classes, gda_means, gda_cov)
        data_1 = data_1 * 200.0
        x_1 = (data_1[:, 0] + center_x1 - new_centers_1[_class][0]) * param_1[_class][1] + new_centers_1[_class][0] + param_1[_class][0][0]
        y_1 = (data_1[:, 1] + center_y1 - new_centers_1[_class][1]) * param_1[_class][1] + new_centers_1[_class][1] + param_1[_class][0][1]
        x_2 = (data_2[:, 0] + center_x2 - new_centers_2[_class][0]) * param_2[_class][1] + new_centers_2[_class][0] + param_2[_class][0][0]
        y_2 = (data_2[:, 1] + center_y2 - new_centers_2[_class][1]) * param_2[_class][1] + new_centers_2[_class][1] + param_2[_class][0][1]
            
        ax1.scatter(x_1, y_1, label=_class, alpha=1.0, s=7, edgecolors='none', zorder=100)
        ax2.scatter(x_2, y_2, label=_class, alpha=1.0, s=7, edgecolors='none', zorder=100)
    
    from matplotlib.patches import Ellipse
    ell1 = Ellipse(xy=(1.0 + param_1["AddToPlaylist"][0][0], -2.0 + param_1["AddToPlaylist"][0][1]), width=6 * param_1["AddToPlaylist"][1], height=10 * param_1["AddToPlaylist"][1], angle=30.0, facecolor='blue', alpha=0.2)
    ax1.add_patch(ell1)
    ell2 = Ellipse(xy=(-20.5 + param_1["BookRestaurant"][0][0], 0.0 + param_1["BookRestaurant"][0][1]), width=4 * param_1["BookRestaurant"][1], height=10 * param_1["BookRestaurant"][1], angle=0.0, facecolor='orange', alpha=0.2)
    ax1.add_patch(ell2)
    ell3 = Ellipse(xy=(11.5 + param_1["RateBook"][0][0], -18.0 + param_1["RateBook"][0][1]), width=3 * param_1["RateBook"][1], height=5 * param_1["RateBook"][1], angle=20.0, facecolor='red', alpha=0.2)
    ax1.add_patch(ell3)
    ell4 = Ellipse(xy=(15.0 + param_1["PlayMusic"][0][0], 6.0 + param_1["PlayMusic"][0][1]), width=3 * param_1["PlayMusic"][1], height=8 * param_1["PlayMusic"][1], angle=45.0, facecolor='green', alpha=0.2)
    ax1.add_patch(ell4)

    ell1 = Ellipse(xy=(0.0 + param_2["AddToPlaylist"][0][0], 0.0 + param_2["AddToPlaylist"][0][1]), width=8 * param_2["AddToPlaylist"][1], height=8 * param_2["AddToPlaylist"][1], angle=0, facecolor='blue', alpha=0.2)
    ax2.add_patch(ell1)
    ell2 = Ellipse(xy=(-18 + param_2["BookRestaurant"][0][0], 0 + param_2["BookRestaurant"][0][1]), width=5 * param_2["BookRestaurant"][1], height=5 * param_2["BookRestaurant"][1], angle=0.0, facecolor='orange', alpha=0.2)
    ax2.add_patch(ell2)
    ell3 = Ellipse(xy=(12.5 + param_2["RateBook"][0][0], -16.0 + param_2["RateBook"][0][1]), width=5 * param_2["RateBook"][1], height=5 * param_2["RateBook"][1], angle=0, facecolor='red', alpha=0.2)
    ax2.add_patch(ell3)
    ell4 = Ellipse(xy=(13.0 + param_2["PlayMusic"][0][0], 16.0 + param_2["PlayMusic"][0][1]), width=5 * param_2["PlayMusic"][1], height=5 * param_2["PlayMusic"][1], angle=0.0, facecolor='green', alpha=0.2)
    ax2.add_patch(ell4)

    ax1.set_xlim(-25, 25)
    ax1.set_xticks([-20, -10, 0, 10, 20])
    ax1.set_ylim(-25, 25)
    ax1.set_yticks([-20, -10, 0, 10, 20])
    ax1.set_aspect(1)
    # ax1.legend()
    ax1.grid(True)
    ax1.set_title("Euclidean Distance Based")

    ax2.set_xlim(-25, 25)
    ax2.set_xticks([-20, -10, 0, 10, 20])
    ax2.set_ylim(-25, 25)
    ax2.set_yticks([-20, -10, 0, 10, 20])
    ax2.set_aspect(1)
    leg = ax1.legend(frameon=True, bbox_to_anchor=(1.00, 0), loc=4, borderaxespad=0, markerscale=1.0, edgecolor='black', fontsize=8)
    ax2.grid(True)
    ax2.set_title("Mahalanobis Distance Based")
    
    plt.savefig(os.path.join(save_dir, "compare_eucli_maha.png"), format="png")
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages(os.path.join(save_dir, "compare_eucli_maha.pdf"))
    pdf.savefig()
    pdf.close()
