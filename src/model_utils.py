
import random
import torch
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import metrics

import src.config as config
from src.model import BertForMultiLabelSequenceClassification


# ENGINE UTILS ###################

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_model(which_model, num_labels, use_checkpoint=False, model_ckpt=None):

    if use_checkpoint and model_ckpt:
        print(f"Loading model checkpoint : {model_ckpt}")
        model_state_dict = torch.load(model_ckpt)
        model = BertForMultiLabelSequenceClassification.from_pretrained(which_model,
                                                                        num_labels=num_labels,
                                                                        state_dict=model_state_dict)
    else:
        if use_checkpoint:
            logging.warning("use_checkpoint set to True, albeit no model_ckpt provided. "
                            "Loading pretrained model without model_state.")
        print(f"Loading pretrained {which_model} for {num_labels} labels")
        model = BertForMultiLabelSequenceClassification.from_pretrained(which_model,
                                                                        num_labels=num_labels)
    return model


def optimizer_params(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    return optimizer_grouped_parameters


def generate_model_filename():
    classifier = config.CLASSIFIER
    date_trained = datetime.today().strftime('%Y%m%d')
    task_name = config.TASK_NAME
    model_version = config.MODEL_VERSION
    suffix = config.COMMENTS

    return config.MODEL_PATH / "{}_{}_{}_{}_{}.bin".format(
            task_name,
            classifier,
            model_version,
            date_trained,
            suffix
    )


# PREDICTION UTILS ###################

def logits_to_discrete(logits, label_list, thr):
    """
        Input:
            - preds: is a DataFrame of shape (n_samples, n_labels) for multilabel
              classification problem. Values of the DataFrame are probabilities
              (logits) yielded from the model i.e. between 0 and 1.
            - labels: list of labels. Need to match the column names of preds
              DataFrame.
            - thr: is either a float (single threshold for all labels) or dict with
              labels as keys and float threshold as values (to have specific thr
              per label).
        Returns:
        - pd.DataFrame of shape (n_samples, n_labels) where
    """
    if isinstance(thr, float):
        single_thr = True
    elif isinstance(thr, dict):
        single_thr = False
    else:
        raise ValueError(f"Type of arg 'thr' needs to be either float or dict. Found {type(thr)}")

    discrete_df = pd.DataFrame()
    for label in label_list:
        _thr = thr if single_thr else thr[label]
        discrete_df[label] = np.array(logits[label] > _thr).astype(int)

    return discrete_df


def get_thr_optimization_params(score_function, independent):

    print("Optimizing logit thresholds for metric: {score_function}")
    if score_function == 'f1':
        thr_opt_func = metrics.f1_score
        lower_better = False
    elif score_function == 'fbeta':
        thr_opt_func = metrics.fbeta_score
        lower_better = False
    elif score_function == 'accuracy':
        thr_opt_func = metrics.accuracy_score
        lower_better = False
    elif score_function == 'precision':
        thr_opt_func = metrics.accuracy_score
        lower_better = False
    elif score_function == 'recall':
        thr_opt_func = metrics.accuracy_score
        lower_better = False
    else:
        raise ValueError("Score Function passed is not recognized. Needs to be one of "
                         "['accuracy', 'precision', 'recall', 'f1', 'fbeta']. "
                         f"Found = {score_function}")

    extra_args = dict()
    if not independent:
        # if optimizing threshold for all labels globally, need to average scores
        # take 'micro' as it takes class imbalances into account
        # other arguments are 'macro', 'weighted', 'samples'
        # check parameters in https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
        extra_args['average'] = 'micro'
        if score_function == 'fbeta':
            extra_args['beta'] = config.FBETA_B
    else:
        extra_args = None

    return thr_opt_func, lower_better, extra_args


def get_optimal_threshold(y_true, y_pred, label_list, eval_func, independent=True, lower_better=False,
                          **eval_fn_kwargs):
    """
        Performs grid search on best performing thresholds from list
        > [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
        using eval_func.

        Input:
            - y_true : DataFrame of one-hot encoded true labels of shape (n_samples, n_labels)
            - y_pred : DataFrame predicted logits of shape (n_samples, n_labels)
            - label_list : list of labels (same order as label columns for y_true and y_pred)
            - eval_func : metric to optimize thresholds against
            - independent: if True, optimizes threshold per label independently. Else, globally
            - lower_better : If lower values are better for eval_func, set lower_better=True
            - eval_fn_kwargs : Extra arguments to be used in eval_func. Example: when optimizing
                               thresholds globally, you would want to pass 'average'=

        Example usage:
           > import src.model.serve as serve
           > from sklearn import metrics
           > logits_train = serve.predict(df_train, model, device, label_list)
           > best_thr, best_scores = get_optimal_threshold(train_df, logits_train,
                                                          label_list,
                                                          metrics.f1_score,
                                                          independent=True,
                                                          lower_better=False,
                                                        #   **{'average':'micro'}
                                                          )

    """
    assert isinstance(y_true, pd.DataFrame) and isinstance(y_pred, pd.DataFrame)
    assert y_true.shape[0] == y_pred.shape[0]

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

    if independent:
        result = {}
        result_eval = {}
        for label in label_list:
            truth = y_true[label]
            preds = y_pred[label]

            best_score = 0
            best_thr = 0
            for thr in thresholds:
                hard_preds = np.array(preds > thr).astype(int)
                score = eval_func(truth, hard_preds, **eval_fn_kwargs)

                if (lower_better and score < best_score) or (
                        not lower_better and score > best_score):
                    best_score = score
                    best_thr = thr

            result[label] = best_thr
            result_eval[label] = best_score

    else:
        print(eval_fn_kwargs)
        best_score = 0
        best_thr = 0
        for thr in thresholds:
            y_pred_discrete = logits_to_discrete(y_pred, label_list, thr)
            score = eval_func(y_true[label_list], y_pred_discrete[label_list], **eval_fn_kwargs)

            if (lower_better and score < best_score) or (not lower_better and score > best_score):
                best_score = score
                best_thr = thr
        result = best_thr
        result_eval = best_score

    return result, result_eval


# CUSTOM METRICS ###################

def roc_auc_score(y_pred, y_true, label_list):
    """ Compute the roc auc score for each class """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i, label in enumerate(label_list):
        fpr[label], tpr[label], _ = metrics.roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[label] = metrics.auc(fpr[label], tpr[label])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    return roc_auc


def fbeta(y_pred: torch.Tensor,
          y_true: torch.Tensor,
          thresh: float = 0.2,
          beta: float = config.FBETA_B,
          eps: float = 1e-9,
          sigmoid: bool = True):
    """ Computes the f_beta between `y_pred` and `y_true` tensors of shape (n, num_labels).

        Usage of beta:
            beta < 0 -> weights more on precision
            beta = 1 -> unweighted f1_score
            beta > 1 -> weights more on recall
        Beta of 0 would only take precision into consideration, and beta +inf would only take recall
        into account.
    """
    beta2 = beta ** 2
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    y_true = y_true.float()
    true_positives = (y_pred * y_true).sum(dim=1)
    precision = true_positives / (y_pred.sum(dim=1) + eps)
    recall = true_positives / (y_true.sum(dim=1) + eps)
    score = (precision * recall) / (precision * beta2 + recall + eps) * (1 + beta2)
    return score.mean().item()


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
