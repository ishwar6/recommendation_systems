import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class EvaluationMetrics:
    def __init__(self, y_true, y_pred):
        """
        Initializes the EvaluationMetrics class with true and predicted labels.
        :param y_true: List or array of true labels.
        :param y_pred: List or array of predicted labels.
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

    def accuracy(self):
        """
        Calculates and returns the accuracy score.
        """
        return accuracy_score(self.y_true, self.y_pred)

    def precision(self):
        """
        Calculates and returns the precision score.
        """
        return precision_score(self.y_true, self.y_pred, average='weighted')

    def recall(self):
        """
        Calculates and returns the recall score.
        """
        return recall_score(self.y_true, self.y_pred, average='weighted')

    def f1(self):
        """
        Calculates and returns the F1 score.
        """
        return f1_score(self.y_true, self.y_pred, average='weighted')

    def summary(self):
        """
        Prints a summary of evaluation metrics.
        """
        print(f'Accuracy: {self.accuracy():.2f}')
        print(f'Precision: {self.precision():.2f}')
        print(f'Recall: {self.recall():.2f}')
        print(f'F1 Score: {self.f1():.2f}')

# Example usage
if __name__ == '__main__':
    true_labels = [0, 1, 0, 1, 0, 1]
    predicted_labels = [0, 1, 1, 1, 0, 0]
    metrics = EvaluationMetrics(true_labels, predicted_labels)
    metrics.summary()