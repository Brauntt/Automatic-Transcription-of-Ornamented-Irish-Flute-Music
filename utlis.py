import json
import torch
import random

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

def calculate_threshold(y_true, y_score):
    fpr, tpr, threshold = roc_curve(y_true, y_score)
    J = tpr - fpr
    thresh = np.argmax(J)
    thresh_value = threshold[thresh]
    return thresh_value