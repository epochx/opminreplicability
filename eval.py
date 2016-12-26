#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple

class ConfusionMatrix(object):

    def __init__(self, true_positives, false_positives, true_negatives, false_negatives):
        self.tp = self.true_positives = true_positives
        self.fp = self.false_positives = false_positives
        self.tn = self.true_negatives = true_negatives
        self.fn = self.false_negatives = false_negatives

    def __repr__(self, *args, **kwargs):
        string = ''
        string += "True Positives: " + str(len(self.tp)) + '\n'
        string += "False Positives: " + str(len(self.fp)) + '\n'
        string += "True Negatives: " + str(len(self.tn)) + '\n'
        string += "False negatives: " + str(len(self.fn)) + '\n'
        return string

    @property
    def precision(self):
        try:
            return 1.0* len(self.tp) / (len(self.tp) + len(self.fp))
        except ZeroDivisionError:
            return 0

    @property
    def recall(self):
        try:
            return 1.0 * len(self.tp) / (len(self.tp) + len(self.fn))
        except ZeroDivisionError:
            return 0

    @property
    def fmeasure(self):
        try:
            precision = self.precision
            recall = self.recall
            return 2 * ((precision * recall) / (precision + recall))
        except ZeroDivisionError:
            return 0

    @property
    def accuracy(self):
        try:
            return 1.0 * (len(self.tp) + len(self.tn)) / (len(self.tp) + len(self.tn) + len(self.fn) + len(self.fp))
        except ZeroDivisionError:
            return 0

    @property
    def measures(self):
        return {"precision":self.precision,
                "recall": self.recall,
                "fmeasure": self.fmeasure}

    p = precision
    r = recall
    a = accuracy
    f = fmeasure


def eval_aspect_ext(model_aspects, corpus_aspects):
    corpus_set = set(corpus_aspects)
    model_set = set(model_aspects)

    tp = corpus_set.intersection(model_set)
    fp = model_set.difference(tp)
    fn = corpus_set.difference(tp)

    return ConfusionMatrix(tp, fp, [], fn)