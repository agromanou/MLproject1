"""
Module description
"""


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
        divisor = (tp + fp)
        if divisor == 0:
            resp = 0
        else:
            resp = tp / divisor
        return resp

    def get_recall(self):
        tp = self.get_true_positives()
        fn = self.get_false_negatives()
        divisor = (tp + fn)
        if divisor == 0:
            resp = 0
        else:
            resp = tp / divisor
        return resp

    def get_f1(self):
        pr = self.get_precision()
        rec = self.get_recall()
        divisor = (pr + rec)
        if divisor == 0:
            resp = 0
        else:
            resp = (2 * pr * rec) / divisor
        return resp

    def get_accuracy(self):
        tp = self.get_true_positives()
        fp = self.get_false_positives()
        tn = self.get_true_negatives()
        fn = self.get_false_negatives()

        divisor = (tp + tn + fp + fn)
        if divisor == 0:
            resp = 0
        else:
            resp = (tp + tn) / divisor
        return resp
