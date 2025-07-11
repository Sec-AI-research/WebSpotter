import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def analyze_location_accuracy(location_predict, location_ground_truth):
    assert len(location_ground_truth) == len(location_predict)
    y_true = np.array(location_ground_truth)
    y_pred = np.array(location_predict)


    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0.0)

    if sum(np.logical_or(y_true, y_pred)) == 0:
        jaccard_index = 0
    else:
        jaccard_index = sum(np.logical_and(y_true, y_pred)) / sum(np.logical_or(y_true, y_pred))

    hamming_temp = len(y_true) - np.count_nonzero(y_true == y_pred)
    hamming = hamming_temp / len(y_true)

    acc = accuracy_score(y_true, y_pred)

    return precision, recall, f1_score, acc, hamming, jaccard_index

class Location_Metric_Recorder():
    def __init__(self):
        self.all_precision = []
        self.all_recall = []
        self.all_f1_scores = []
        self.all_acc = []
        self.all_hamming = []
        self.all_jaccard_index = []
    
    def append(self, precision, recall, f1_score, acc, hamming, jaccard_index):
        self.all_precision.append(precision)
        self.all_recall.append(recall)
        self.all_f1_scores.append(f1_score)
        self.all_acc.append(acc)
        self.all_hamming.append(hamming)
        self.all_jaccard_index.append(jaccard_index)
    
    def get_avg(self):
        '''
        Return: dict, average of all metrics, keys in ['precision', 'recall', 'f1_score', 'acc', 'hamming', 'jaccard_index']
        '''
        number = len(self.all_precision)
        return {'precision': sum(self.all_precision) / number, 'recall': sum(self.all_recall) / number, 'f1_score': sum(self.all_f1_scores) / number, 'acc': sum(self.all_acc) / number, 'hamming': sum(self.all_hamming) / number, 'jaccard_index': sum(self.all_jaccard_index) / number}
