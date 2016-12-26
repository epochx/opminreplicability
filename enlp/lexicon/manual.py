#!/usr/bin/python
# -*- coding: utf-8 -*-


class ManualLexicon(dict):

    def __init__(self, name):
        super(ManualLexicon, self).__init__(self)
        self.name = name

    def __str__(self):
        return self.name

    def __setitem__(self, key, orientation):
        super(ManualLexicon, self).__setitem__(key, LexiconWord(key, orientation))


class LexiconWord(object):

    def __init__(self, string, orientation):
        self.string = string
        self.orientation = orientation
