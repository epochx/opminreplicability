#!/usr/bin/python
# -*- coding: utf-8 -*-

# --- RELATION --------------------------------------------------------------


class Relation(object):

    def __init__(self, head, label, dependent):
        self.head = head
        self.label = label
        self.dependent = dependent

    def __repr__(self, *args, **kwargs):
        return "{0}({1},{2})".format(self.label,
                                     self.head.string,
                                     self.dependent.string)

    def __contains__(self, token):
        if token in [self.head, self.dependent]:
            return True
        return False

    def is_head(self, token):
        if token is self.head:
            return True
        return False
