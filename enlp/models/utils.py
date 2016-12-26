#!/usr/bin/python
# -*- coding: utf-8 -*-

# classes for Token keys


class LowerStringKey(object):

    def init(self):
        pass

    def __call__(self, token):
        return token.string.lower()

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.__class__.__name__


class StemKey(object):

    def init(self):
        pass

    def __call__(self, token):
        if token.stem:
            return token.stem
        else:
            return token.string.lower()

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.__class__.__name__
