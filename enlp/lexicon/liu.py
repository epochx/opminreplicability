#!/usr/bin/python
# -*- coding: utf-8 -*-

from os import path
from ..settings import CORPORA_PATH
OPINION_LEXICON_PATH = path.join(CORPORA_PATH, 'lexicon')


class LexiconWord(object):

    def __init__(self, string, orientation):
        self.string = string
        self.orientation = orientation


class OpinionLexicon(dict):

    filepath = ''

    def __init__(self):
        pos_filepath = path.join(OPINION_LEXICON_PATH,
                                 self.filepath,
                                 "positive_words.txt")
        neg_filepath = path.join(OPINION_LEXICON_PATH,
                                 self.filepath,
                                 "negative_words.txt")

        pos = open(pos_filepath, "r")
        neg = open(neg_filepath, "r")

        for line in pos.readlines():
            string = str(line).replace("\n", "")
            self[string] = LexiconWord(string, 1.0)

        for line in neg.readlines():
            string = str(line).replace("\n", "")
            self[string] = LexiconWord(string, -1.0)

    @property
    def name(self):
        return str(self)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.__class__.__name__

    @property
    def positive(self):
        return {key: value
                for key, value in self.iteritems()
                if value == 1}

    @property
    def negative(self):
        return {key: value
                for key, value in self.iteritems()
                if value == -1}

LiuLexicon = type("LiuLexicon",
                  (OpinionLexicon,),
                  {"filepath": "liu"})