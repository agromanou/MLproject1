"""
Module description
"""

import numpy as np

class Evaluation:

    def __init__(self, y_actual, y_pred):
        self.y_actual = y_actual.tolist()
        self.y_pred = y_pred.tolist()

    
    def get_count(self, target):
        count = 0
        for t in zip(self.y_actual, self.y_pred):
            if t == target:
                count += 1
        return count

    def get_true_positives(self):
        return self.get_count((1, 1))
    
    def get_false_positives(self):
        return self.get_count((0, 1))
    
    def get_true_negatives(self):
        return self.get_count((0, 0))

    def get_false_negatives(self):
        return self.get_count((1, 0))
    
    def get_precision(self):
        tp = self.get_true_positives()
        fp = self.get_false_positives()
        return tp / (tp + fp)

    def get_recall(self):
        tp = self.get_true_positives()
        fn = self.get_false_negatives()
        return tp / (tp + fn)

    def get_f1(self):
        pr = self.get_precision()
        rec = self.get_recall()
        return 2 * pr * rec / (pr + rec)
    
    def get_accuracy(self):
        tp = self.get_true_positives()
        fp = self.get_false_positives()
        tn = self.get_true_negatives()
        fn = self.get_false_negatives()
        return (tp + tn) / (tp + tn + fp + fn)
