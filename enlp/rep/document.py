#!/usr/bin/python
# -*- coding: utf-8 -*-

from .utils import _uid


class Document(object):

    def __init__(self,  id=None):
        self.id = id if id else _uid()
        self.sentences = []

    def expand(self, sentences):
        for sentence in sentences:
            self.append(sentence)

    def append(self, s):
        s._document = self
        self.sentences.append(s)
