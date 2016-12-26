#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils import encode_entities, decode_entities

# -- TOKEN ------------------------------------------------------------------


class Token(object):

    def __init__(self, sentence, string,
                 start=None, end=None,
                 lemma=None, pos_tag=None,
                 iob_tag=None, chunk=None, index=0):
        """ A token in a sentence.
            - sentence: the sentence object the Token belongs to 
            - lemma: base form of the Token; "was" => "be".
            - pos_tag: the part-of-speech tag of the Token; "NN" => a noun.
            - iob_tag: the chunk tag of the Token.
            - chunk: the chunk (or phrase) this Token belongs to.
            - index: the index (position) of the Token in the sentence.
        """
        if not isinstance(string, unicode):
            try:
                string = string.decode("utf-8")
            except:
                pass
        self.sentence = sentence
        self.index = index
        self.string = string
        self.start = start
        self.end = end
        self._lemma = lemma
        self._type = pos_tag
        self._iob = iob_tag
        self.chunk = chunk

    @property
    def is_lemmatized(self):
        return True if self.lemma else False

    @property
    def is_tagged(self):
        return True if self.pos else False

    @property
    def is_chunked(self):
        return True if self.iob else False

    @property
    def relations(self):
        return [relation for relation in self.sentence.relations
                if self in relation]

    @property
    def head(self):
        """
        Returns the head token and the relation label
        as a tuple.
        """
        heads = [(relation.head, relation.label)
                 for relation in self.sentence.relations
                 if relation.dependent == self]
        if len(heads) > 0:
            if len(heads) > 1:
                pass
                # print "Two heads:"
                # print self
                # print heads
            return heads[0]
        # print "No head"
        # print self
        return None

    @property
    def dependents(self):
        """
        Returns the dependent tokens and their
        relation labels as a list of tuples.
        """
        return [(relation.dependent, relation.label)
                for relation in self.sentence.relations
                if relation.head == self]

    def copy(self, chunk=None):
        w = Token(self.sentence,
                  self.string,
                  lemma=self._lemma,
                  pos_tag=self._type,
                  iob_tag=self._iob,
                  chunk=self.chunk,
                  index=self.index)
        return w

    def _get_pos_tag(self):
        return self._type

    def _set_pos_tag(self, pos_tag):
        self._type = pos_tag

    POS = pos = pos_tag = tag = part_of_speech = property(_get_pos_tag, _set_pos_tag)

    def _get_iob_tag(self):
        return self._iob

    def _set_iob_tag(self, iob_tag):
        self._iob = iob_tag

    IOB = iob_tag = iob = property(_get_iob_tag, _set_iob_tag)

    def _set_lemma(self, lemma):
        self._lemma = lemma

    def _get_lemma(self):
        return self._lemma

    STEM = LEMMA = lemma = stem = property(_get_lemma, _set_lemma)

    def next(self, pos_tag=None):
        """ Returns the next Token in the sentence with the given type.
        """
        i = self.index + 1
        s = self.sentence
        while i < len(s):
            if pos_tag in (s[i].pos, None):
                return s[i]
            i += 1

    def previous(self, pos_tag=None):
        """ Returns the next previous Token in the sentence with the given type.
        """
        i = self.index - 1
        s = self.sentence
        while i > 0:
            if pos_tag in (s[i].pos, None):
                return s[i]
            i -= 1

    # Token.string and unicode(Token) are Unicode strings.
    # repr(Token) is a Python string (with Unicode characters encoded).
    def __unicode__(self):
        return self.string

    def __repr__(self):
        if self.is_tagged:
            return "Token(%s)" % repr("%s/%s" % (
                encode_entities(self.string),
                self.pos is not None and self.pos))
        else:
            return "Token(%s)" % self.string

    def __eq__(self, token):
        return id(self) == id(token)

    def __ne__(self, token):
        return id(self) != id(token)