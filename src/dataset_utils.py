
import nltk
import logging
import pandas as pd
import numpy as np
import src.config as config
from sklearn.preprocessing import MultiLabelBinarizer

nltk.download('punkt')  # necessary for nltk.sent_tokenize


def multi_label_binarize(df):
    proc_df = df[['description', 'target']]

    mlb = MultiLabelBinarizer()
    popped_classes = pd.DataFrame(
            mlb.fit_transform(proc_df.pop('target')),
            columns=mlb.classes_,
            index=proc_df.index
    )

    proc_df = proc_df.join(popped_classes).drop_duplicates()
    return proc_df


def get_description_length(description):
    words = []
    for sentence in nltk.sent_tokenize(description):
        words = words + nltk.word_tokenize(sentence)
    return len(words)


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()