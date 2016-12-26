#!/usr/bin/python
# -*- coding: utf-8 -*-

from .utils import BEGIN, INSIDE, OUTSIDE

# --- CHUNK ------------------------------------------------------------------


class Chunk(object):

    def __init__(self, sentence, tokens=[], chunk_tag=None):
        """ A list of Tokens that make up a phrase in the sentence.
            - chunk_tag: the phrase tag; "NP"
              => a noun phrase (e.g., "the black cat").
        """
        self.sentence = sentence
        self.tokens = []
        self.type = chunk_tag  # NP, VP, ADJP ...
        self.extend(tokens)

    def extend(self, tokens):
        for token in tokens:
            self.append(token)

    def append(self, token):
        self.tokens.append(token)
        token.chunk = self

    def __getitem__(self, index):
        return self.tokens[index]

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return self.tokens.__iter__()

    def _get_tag(self):
        return self.type

    def _set_tag(self, v):
        self.type = v

    tag = pos = pos_tag = part_of_speech = property(_get_tag, _set_tag)

    @property
    def start(self):
        return self.tokens[0].index

    @property
    def stop(self):
        return self.tokens[-1].index + 1

    @property
    def range(self):
        return range(self.start, self.stop)

    @property
    def span(self):
        return (self.start, self.stop)

    index = span

    @property
    def lemmata(self):
        return [token.lemma for token in self.tokens]

    @property
    def tagged(self):
        return [(token.string, token.pos) for token in self.token]

    def nearest(self, type="NP"):
        """ Returns the nearest chunk in the sentence with the given type.
            This can be used (for example) to find adverbs and adjectives related to verbs,
            as in: "the cat is ravenous" => is what? => "ravenous".
        """
        candidate, d = None, len(self.sentence.chunks)
        i = self.sentence.chunks.index(self)
        for j, chunk in enumerate(self.sentence.chunks):
            if chunk.type.startswith(type) and abs(i-j) < d:
                candidate, d = chunk, abs(i-j)
        return candidate

    def next(self, type=None):
        """ Returns the next chunk in the sentence with the given type.
        """
        i = self.stop
        s = self.sentence
        while i < len(s):
            if s[i].chunk is not None and type in (s[i].chunk.type, None):
                return s[i].chunk
            i += 1

    def previous(self, type=None):
        """ Returns the next previous chunk in the sentence with the given type.
        """
        i = self.start - 1
        s = self.sentence
        while i > 0:
            if s[i].chunk is not None and type in (s[i].chunk.type, None):
                return s[i].chunk
            i -= 1

    # Chunk.string and unicode(Chunk) are Unicode strings.
    # repr(Chunk) is a Python string (with Unicode characters encoded).

    @property
    def string(self):
        return u" ".join(token.string for token in self.tokens)

    def __unicode__(self):
        return self.string

    def __repr__(self):
        return "Chunk(%s)" % repr("%s/%s") % (
                self.string,
                self.type is not None and self.type or OUTSIDE)

    def __eq__(self, chunk):
        return id(self) == id(chunk)

    def __ne__(self, chunk):
        return id(self) != id(chunk)
