#!/usr/bin/python
# -*- coding: utf-8 -*-

_UID = 0


def _uid():
    global _UID
    _UID += 1
    return _UID

# B- marks the start of a chunk: the/DT/B-NP cat/NN/I-NP
# I- words are inside a chunk.
# O- words are outside a chunk (punctuation etc.).
BEGIN, INSIDE, OUTSIDE = 'B', 'I', 'O'

# The output of parse() is a slash-formatted string (e.g., "the/DT cat/NN"),
# so slashes in words themselves are encoded as &slash;
SLASH = "&slash;"

encode_entities = lambda string: string.replace("/", SLASH)
decode_entities = lambda string: string.replace(SLASH, "/")
