#!/usr/bin/python
# -*- coding: utf-8 -*-

from .stopwords import StanfordStopwords
import re
import math
from collections import defaultdict


# ------- MAIN FUNCTION -------------------------------------------------------

def generate_multi_word_terms(sentences, method='ngram', limit=0, **kwargs):
    """
    Extracts the a set multi word terms from a set ot sentences using the Cvalue
    technique, from "Automatic Recognition of Multi-Word Terms: the C-value/NC
    -value Method" by Katerina Frantziy, Sophia Ananiadouy, Hideki Mimaz

    Input:
        - sentences        : list of sentences
        - limit            : maximum number of terms to extract, default=0
                             (no limit)
        - method           : string, "ngram" or "condition_#, default=ngram
        - stopwords        : iterable of strings, stopwords/stoplist
        - max_ngram_len    : int, maximum length of ngram to use as candidate
                             for multiword term when using method "ngram",
                             default=4
        - key              : function to apply to every token in each sentence,
                             default=x: x.string
    Output
        - list of the extracted terms, as strings.
    """
    assert method in ["ngram", "condition_1",
                      "condition_2", "condition_3"]

    method = kwargs.get("method", "condition_2")
    key = kwargs.get("key", lambda x: x.string)
    stopwords = kwargs.get("stopwords", StanfordStopwords())
    max_ngram_len = kwargs.get("max_ngram_len", 4)

    if method == "ngram":
        word_groups = _get_ngram_word_groups(sentences,
                                             max_ngram_len=max_ngram_len,
                                             stoplist=stopwords, key=key)

    elif "condition" in method:
        word_groups = _get_cond_word_groups(sentences, condition=method,
                                            stoplist=stopwords, key=key)

    limited = _limit_word_groups(word_groups, limit=limit)

    result = []
    for g in limited:
        result += g.original

    return result


# ------- CLASSES -------------------------------------------------------------

class WordGroup(object):

    def __init__(self, groups, string, original):
        self.groups = groups
        self.string = string
        self.freq = 1
        self.original = [original]
        self.t = len(string.split(" "))

    def add(self, original):
        if original not in self.original:
            self.original.append(original)
        self.freq += 1

    @property
    def super_sets(self):
        """Supersets"""
        return [group for group in self.groups.values()
                if group is not self and self.string in group.string]

    @property
    def has_super(self):
        """
        Returns true if there is any word group that is super set of this one
        """
        return True if len(self.super_sets) else False

    @property
    def cvalue(self):
        """
        If wordgroup t is not contained by any other terms:
            Cvalue(t) = log(|t|)× frq(t)
        Otherwise:
            Cvalue(t) = log(|t|) × 1/N(L) * ( frq(t) - sum [ frq(l) for l in L]

        Where:
            - |t| denotes the number of words contained by t
            - frq(t) indicates the frequency of occurrence of t in the corpus
            - L is the set of multi-word terms containing t
            - n(L) denotes the number of terms in S
        """
        cvalue = math.log(self.t)*self.freq
        if self.has_super:
            cvalue = cvalue * 1/len(self.super_sets)*sum([group.t
                                                          for group in self.super_sets])
        return cvalue

# ------- FUNCTIONS -----------------------------------------------------------

# Conditions for grouping terms, as given in:
# Automatic Recognition of Multi-Word Terms: the C-value/NC-value Method
# by Katerina Frantziy, Sophia Ananiadouy, Hideki Mimaz

condition_1 = re.compile(r"""(\d*-NN\D?\s)+(\d*-NN\D?\s?)""")

condition_2 = re.compile(r"""(\d*-JJ\D?\s|\d*-NN\D?\s)+(\d*-NN\D?\s?)""")

condition_3 = re.compile(r"""((\d*-JJ\D?\s|\d*-NN\D?\s)+(\d*-NN\D?\s?))|
                             (((\d*-JJ\D?\s|\d*-NN\D?\s)*(\d*-IN\s)?)
                             (\d*-JJ\D?\s|\d*-NN\D?\s)*(\d*-NN\D?\s?))""")


def condition_matches(sentence, condition):
    tags = " ".join(["{0}-{1}".format(t.index, t.pos)
                     for t in sentence])
    for match in re.finditer(condition, tags):
        g = match.group().strip().split()
        s_index, s_pos = g[0].split("-")
        e_index, e_pos = g[-1].split("-")
        yield int(s_index), int(e_index)+1


def _generate_sentence_ngrams(input_list, n):
    """Generates ngrams of a maximum length of n, from a list"""
    return zip(*[input_list[i:] for i in range(n)])


def _get_stoplist(sentences, percentage=10, key=lambda x: x.string):
    """
    Extract a stop-list as the top frequent words in the corpus. A stop-list
    is a list of words which are not expected to occur as term words in
    that domain

    Input:
        - sentences  : list of sentences
        - percentage : the % of top frequent terms to be extracted ad stoplist,
                       default=10
        - key        : function to apply to each token in each sentence before
                       added to the global list of words used to extract the
                       stoplist
    Output:
        - stoplist: list of strings
    """
    # get word frequency
    words = defaultdict(int)
    for sentence in sentences:
        for token in sentence:
            words[key(token)] += 1
    words = sorted(words.items(), key=lambda x: x[1], reverse=True)
    stoplist =[word for word, value
               in words[:int(round(len(words)/percentage, 0))]]
    return stoplist


def _get_cond_word_groups(sentences, condition="condition_2",
                          stoplist=StanfordStopwords(),
                          key=lambda x: x.string):
    """
    Get word groups based on the given conditions
    Input:
        - sentences : list of sentences
        - condition : string, condition_#, condition to extract a word group
                      from a sentence, default=condition_2
        - stoplist  : list of strings. Groups containing any words in the list
                      are not extracted, default=stanford_stopwords
        - key       : function to apply to each token before extraction

    Output:
        - groups: dict of WordGroups
    """
    if condition == "condition_1":
        real_condition = condition_1

    if condition == "condition_2":
        real_condition = condition_2

    if condition == "condition_3":
        real_condition = condition_3

    groups = {}
    for sentence in sentences:
        for i, j in condition_matches(sentence, real_condition):
            if not any([token.string.lower() in stoplist
                        for token in sentence.tokens[i:j]]):
                original = " ".join([token.string
                                     for token in sentence.tokens[i:j]])
                string = " ".join([key(token)
                                   for token in sentence.tokens[i:j]])
                g = groups.get(string, None)
                if g:
                    g.add(original)
                else:
                    g = WordGroup(groups, string, original)
                    groups[string] = g
    return groups


def _get_ngram_word_groups(sentences, max_ngram_len=4,
                           stoplist=StanfordStopwords(),
                           key=lambda x: x.string):
    """
    Get word groups based on ngrams.

    Input:
        - sentences     : list of sentences
        - max_ngram_len : max length of ngrams to extract (min is 2), default=4
        - stoplist      : list of strings. Groups containing any words in the list
                          are not extracted, default=stanford_stopwords
        - key           : function to apply to each token before group extraction
                          default=lambda x: x.string

    Output:
        - groups: dict of WordGroups
    """
    assert max_ngram_len > 1

    groups = {}
    for sentence in sentences:
        for n in range(2, max_ngram_len+1):
            for ngram in _generate_sentence_ngrams(sentence, n):
                if not any([token.string.lower() in stoplist
                            for token in ngram]):
                    original = " ".join([token.string for token in ngram])
                    string = " ".join([key(token) for token in ngram])
                    g = groups.get(string, None)
                    if g:
                        g.add(original)
                    else:
                        g = WordGroup(groups, string, original)
                        groups[string] = g
    return groups


def _limit_word_groups(groups, limit=0):
    """
    Outputs a list of word groups
    Input:
        -groups: dict of WordGroups
    Output:
        - list of selected WordGroups
    """
    ordered_groups = sorted(groups.values(),
                            key=lambda g: g.cvalue,
                            reverse=True)
    if limit:
        result = ordered_groups[:int(limit)]
        return result
    return ordered_groups
