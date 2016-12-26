#!/usr/bin/python
# -*- coding: utf-8 -*-

from os import path
from .settings import CORPORA_PATH

STOPWORDS_PATH = path.join(CORPORA_PATH, 'stopwords')


class Stopwords(set):

    filepath = ""

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    def __init__(self):
        """ Returns a set of stopwords from a file"""
        stopwords_file = open(self.filepath, "r")
        for line in stopwords_file.readlines():
            line2 = line.replace("\n", "")  
            self.add(line2)

NLTKStopwords = type("NLTKStopwords",
                     (Stopwords,),
                     {"filepath": path.join(STOPWORDS_PATH,
                                            "nltk_stopwords.txt")})

StanfordStopwords = type("StanfordStopwords",
                         (Stopwords,),
                         {"filepath": path.join(STOPWORDS_PATH,
                                                "stanford_stopwords.txt")})
