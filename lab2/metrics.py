import numpy as np
import sklearn.metrics as skm


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes = None) -> np.ndarray:
    """"
    Computes the confusion matrix from labels (y_true) and predictions (y_pred).
    The matrix columns represent the prediction labels and the rows represent the ground truth labels.
    The confusion matrix is always a 2-D array of shape `[num_classes, num_classes]`,
    where `num_classes` is the number of valid labels for a given classification task.
    The arguments y_true and y_pred must have the same shapes in order for this function to work

    num_classes represents the number of classes for the classification problem. If this is not provided,
    it will be computed from both y_true and y_pred
    """
    conf_mat = None
    # TODO your code here - compute the confusion matrix
    # even here try to use vectorization, so NO for loops

    # 0. if the number of classes is not provided, compute it based on the y_true and y_pred arrays
    if num_classes is None:
        num_classes = len(set(y_true))

    # 1. create a confusion matrix of shape (num_classes, num_classes) and initialize it to 0
    conf_mat = np.zeros((num_classes, num_classes))

    # 2. use argmax to get the maximal prediction for each sample
    np.add.at(conf_mat, (y_true, y_pred), 1)

    # hint: you might find np.add.at useful: https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html

    # end TODO your code here
    return conf_mat


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes=None) -> float:
    """"
    Computes the precision score.
    For binary classification, the precision score is defined as the ratio tp / (tp + fp)
    where tp is the number of true positives and fp the number of false positives.

    For multiclass classification, the precision and recall scores are obtained by summing over the rows / columns
    of the confusion matrix.

    num_classes represents the number of classes for the classification problem. If this is not provided,
    it will be computed from both y_true and y_pred
    """
    precision = 0
    # TODO your code here
    confusion_matrix = compute_confusion_matrix(y_true, y_pred)
    rows_sum = np.sum(confusion_matrix, axis=0)
    precision = np.diagonal(confusion_matrix / rows_sum)

    # end TODO your code here
    return precision


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes=None)  -> float:
    """"
    Computes the recall score.
    For binary classification, the recall score is defined as the ratio tp / (tp + fn)
    where tp is the number of true positives and fn the number of false negatives

    For multiclass classification, the precision and recall scores are obtained by summing over the rows / columns
    of the confusion matrix.

    num_classes represents the number of classes for the classification problem. If this is not provided,
    it will be computed from both y_true and y_pred
    """
    recall = None
    # TODO your code here
    confusion_matrix = compute_confusion_matrix(y_true, y_pred)
    columns_sum = np.sum(confusion_matrix, axis=1)
    recall = np.diagonal(confusion_matrix / columns_sum)

    # end TODO your code here
    return recall


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    acc_score = 0
    # TODO your code here
    # remember, use vectorization, so no for loops
    # hint: you might find np.trace useful here https://numpy.org/doc/stable/reference/generated/numpy.trace.html
    confusion_matrix = compute_confusion_matrix(y_true, y_pred)
    confusion_matrix_trace = np.trace(confusion_matrix)
    confusion_matrix_total_sum = np.sum(confusion_matrix)
    acc_score = confusion_matrix_trace / confusion_matrix_total_sum 

    # end TODO your code here
    return acc_score


if __name__ == '__main__':
    pass
    # TODO your tests here
    # add some test for your code.
    # you could use the sklean.metrics module (with macro averaging to check your results)
    y_true = [2, 1, 3, 1, 4, 5, 6, 0, 8, 0, 9, 7, 1, 4, 2, 6, 4]
    y_pred = [2, 1, 9, 1, 4, 5, 6, 0, 3, 0, 9, 7, 1, 4, 1, 6, 8]
    confusion_matrix = compute_confusion_matrix(y_true, y_pred)

    skm_confusion_matrix = skm.confusion_matrix(y_true, y_pred)
    assert (np.allclose(confusion_matrix, skm_confusion_matrix))

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    skm_accuracy = skm.accuracy_score(y_true, y_pred)
    skm_precision = skm.precision_score(y_true, y_pred, average=None)
    skm_recall = skm.recall_score(y_true, y_pred, average=None)

    assert(np.allclose(accuracy, skm_accuracy))
    assert(np.allclose(precision, skm_precision))
    assert(np.allclose(recall, skm_recall))