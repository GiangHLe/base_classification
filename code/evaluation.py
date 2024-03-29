import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

'''
Queue:
    + AUROC
    + Confusion matrix
Done:
    + accuracy
    + precision_recall_f1
'''

def array2tensor(array, squeeze = True):
    if isinstance(array, np.ndarray):
        array = array.squeeze() if squeeze else array
        return array.astype(np.float32)
    else:
        array = array.detach().cpu().float().numpy()
        return array.squeeze() if squeeze else array

def check_tensor(tensor1, tensor2):
    return [array2tensor(tensor1), array2tensor(tensor2)]

def accuracy(y_pred, y_true, threshold = 0.5, binary = False):
    '''
    Return the accuracy
    '''
    y_pred, y_true = check_tensor(y_pred, y_true)
    if binary:
        y_true = y_true.astype(np.int32)
        y_pred = (y_pred > threshold).astype(np.int32)
    return accuracy_score(y_true, y_pred, normalize=True)

def precision_recall_f1(y_pred, y_true, average = 'macro'):
    '''
    Return the precision score with average mode:
        + macro: compute precision score of each class then take average (default)
        + micro: determine all True Positive and all False Positive elements then compute the precision
    '''
    y_pred, y_true = check_tensor(y_pred, y_true)
    return [
        precision_score(y_pred, y_true, average = average),
        recall_score(y_pred, y_true, average = average),
        f1_score(y_pred, y_true, average = average)
    ]

