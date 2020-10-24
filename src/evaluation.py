
class Evaluation:
    """
    This class is responsible for the evaluation of a given predicted labels
    compared with the actual ones.
    """

    def __init__(self, y_actual, y_pred):
        """
        :param y_actual: np.array with the actual labels
        :param y_pred: np.array with the predicted labels
        """
        self.y_actual = y_actual.tolist()
        self.y_pred = y_pred.tolist()

    def get_true_positives(self):
        """ Count true positives """
        return self._get_count((1, 1))

    def get_false_positives(self):
        """ Count false positives """
        return self._get_count((0, 1))

    def get_true_negatives(self):
        """ Count true negatives """
        return self._get_count((0, 0))

    def get_false_negatives(self):
        """ Count false negatives """
        return self._get_count((1, 0))

    def get_precision(self):
        """
        Calculate precision:
            precision = true positives / (true positives + false positives)
        """
        tp = self.get_true_positives()
        fp = self.get_false_positives()
        divisor = (tp + fp)
        if divisor == 0:
            resp = 0
        else:
            resp = tp / divisor
        return resp

    def get_recall(self):
        """
        Calculate recall:
            recall = true positives / (true positives + false negatives)
        """
        tp = self.get_true_positives()
        fn = self.get_false_negatives()
        divisor = (tp + fn)
        if divisor == 0:
            resp = 0
        else:
            resp = tp / divisor
        return resp

    def get_f1(self):
        """
        Compute F1 score:
            precision = 2 * (precision * recall) / (precision + recall)
        """
        pr = self.get_precision()
        rec = self.get_recall()
        divisor = (pr + rec)
        if divisor == 0:
            resp = 0
        else:
            resp = (2 * pr * rec) / divisor
        return resp

    def get_accuracy(self):
        """
        Calculate accuracy:
            accuracy = (true positives + true negatives) / all predictions
        """
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

    def _get_count(self, target):
        """
        Compare each actual-pred tuple with the given tuple of labels.

        :param target: tuple of labels
        :return: number of matching tuples in actual-pred labels
        """
        count = 0
        for t in zip(self.y_actual, self.y_pred):
            if t == target:
                count += 1
        return count
