#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import warnings

from ..rep import Sentence
from ..corenlp import CoreNLP

class Stemmer():

    def __str__(self):
        return self.__class__.__name__

    def stem(self):
        raise NotImplementedError

    def batch_stem(self):
        raise NotImplementedError

# ------- PORTER STEMMER -----------------------------------------------------

# pyporter2: An implementation of the Porter2 stemming algorithm.
# See http://snowball.tartarus.org/algorithms/english/stemmer.html"""

# Copyright (c) 2008 Michael Dirolf (mike at dirolf dot com)

# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

regexp = re.compile(r"[^aeiouy]*[aeiouy]+[^aeiouy](\w*)")


def get_r1(word):
    # exceptional forms
    if word.startswith('gener') or word.startswith('arsen'):
        return 5
    if word.startswith('commun'):
        return 6

    # normal form
    match = regexp.match(word)
    if match:
        return match.start(1)
    return len(word)


def get_r2(word):
    match = regexp.match(word, get_r1(word))
    if match:
        return match.start(1)
    return len(word)


def ends_with_short_syllable(word):
    if len(word) == 2:
        if re.match(r"^[aeiouy][^aeiouy]$", word):
            return True
    if re.match(r".*[^aeiouy][aeiouy][^aeiouywxY]$", word):
        return True
    return False


def is_short_word(word):
    if ends_with_short_syllable(word):
        if get_r1(word) == len(word):
            return True
    return False


def remove_initial_apostrophe(word):
    if word.startswith("'"):
        return word[1:]
    return word


def capitalize_consonant_ys(word):
    if word.startswith('y'):
        word = 'Y' + word[1:]
    return re.sub(r"([aeiouy])y", '\g<1>Y', word)


def step_0(word):
    if word.endswith("'s'"):
        return word[:-3]
    if word.endswith("'s"):
        return word[:-2]
    if word.endswith("'"):
        return word[:-1]
    return word


def step_1a(word):
    if word.endswith('sses'):
        return word[:-4] + 'ss'
    if word.endswith('ied') or word.endswith('ies'):
        if len(word) > 4:
            return word[:-3] + 'i'
        else:
            return word[:-3] + 'ie'
    if word.endswith('us') or word.endswith('ss'):
        return word
    if word.endswith('s'):
        preceding = word[:-1]
        if re.search(r"[aeiouy].", preceding):
            return preceding
        return word
    return word


def step_1b(word, r1):
    if word.endswith('eedly'):
        if len(word) - 5 >= r1:
            return word[:-3]
        return word
    if word.endswith('eed'):
        if len(word) - 3 >= r1:
            return word[:-1]
        return word

    def ends_with_double(word):
        doubles = ['bb', 'dd', 'ff', 'gg', 'mm', 'nn', 'pp', 'rr', 'tt']
        for double in doubles:
            if word.endswith(double):
                return True
        return False

    def step_1b_helper(word):
        if word.endswith('at') or word.endswith('bl') or word.endswith('iz'):
            return word + 'e'
        if ends_with_double(word):
            return word[:-1]
        if is_short_word(word):
            return word + 'e'
        return word

    suffixes = ['ed', 'edly', 'ing', 'ingly']
    for suffix in suffixes:
        if word.endswith(suffix):
            preceding = word[:-len(suffix)]
            if re.search(r"[aeiouy]", preceding):
                return step_1b_helper(preceding)
            return word

    return word


def step_1c(word):
    if word.endswith('y') or word.endswith('Y'):
        if word[-2] not in 'aeiouy':
            if len(word) > 2:
                return word[:-1] + 'i'
    return word


def step_2(word, r1):
    def step_2_helper(end, repl, prev):
        if word.endswith(end):
            if len(word) - len(end) >= r1:
                if prev == []:
                    return word[:-len(end)] + repl
                for p in prev:
                    if word[:-len(end)].endswith(p):
                        return word[:-len(end)] + repl
            return word
        return None

    triples = [('ization', 'ize', []),
               ('ational', 'ate', []),
               ('fulness', 'ful', []),
               ('ousness', 'ous', []),
               ('iveness', 'ive', []),
               ('tional', 'tion', []),
               ('biliti', 'ble', []),
               ('lessli', 'less', []),
               ('entli', 'ent', []),
               ('ation', 'ate', []),
               ('alism', 'al', []),
               ('aliti', 'al', []),
               ('ousli', 'ous', []),
               ('iviti', 'ive', []),
               ('fulli', 'ful', []),
               ('enci', 'ence', []),
               ('anci', 'ance', []),
               ('abli', 'able', []),
               ('izer', 'ize', []),
               ('ator', 'ate', []),
               ('alli', 'al', []),
               ('bli', 'ble', []),
               ('ogi', 'og', ['l']),
               ('li', '', ['c', 'd', 'e', 'g', 'h', 'k', 'm', 'n', 'r', 't'])]

    for trip in triples:
        attempt = step_2_helper(trip[0], trip[1], trip[2])
        if attempt:
            return attempt

    return word


def step_3(word, r1, r2):
    def step_3_helper(end, repl, r2_necessary):
        if word.endswith(end):
            if len(word) - len(end) >= r1:
                if not r2_necessary:
                    return word[:-len(end)] + repl
                else:
                    if len(word) - len(end) >= r2:
                        return word[:-len(end)] + repl
            return word
        return None

    triples = [('ational', 'ate', False),
               ('tional', 'tion', False),
               ('alize', 'al', False),
               ('icate', 'ic', False),
               ('iciti', 'ic', False),
               ('ative', '', True),
               ('ical', 'ic', False),
               ('ness', '', False),
               ('ful', '', False)]

    for trip in triples:
        attempt = step_3_helper(trip[0], trip[1], trip[2])
        if attempt:
            return attempt

    return word


def step_4(word, r2):
    delete_list = ['al', 'ance', 'ence', 'er', 'ic', 'able', 'ible', 'ant', 'ement',
                   'ment', 'ent', 'ism', 'ate', 'iti', 'ous', 'ive', 'ize']

    for end in delete_list:
        if word.endswith(end):
            if len(word) - len(end) >= r2:
                return word[:-len(end)]
            return word

    if word.endswith('sion') or word.endswith('tion'):
        if len(word) - 3 >= r2:
            return word[:-3]

    return word


def step_5(word, r1, r2):
    if word.endswith('l'):
        if len(word) - 1 >= r2 and word[-2] == 'l':
            return word[:-1]
        return word

    if word.endswith('e'):
        if len(word) - 1 >= r2:
            return word[:-1]
        if len(word) - 1 >= r1 and not ends_with_short_syllable(word[:-1]):
            return word[:-1]

    return word


def normalize_ys(word):
    return word.replace('Y', 'y')

exceptional_forms = {'skis': 'ski',
                     'skies': 'sky',
                     'dying': 'die',
                     'lying': 'lie',
                     'tying': 'tie',
                     'idly': 'idl',
                     'gently': 'gentl',
                     'ugly': 'ugli',
                     'early': 'earli',
                     'only': 'onli',
                     'singly': 'singl',
                     'sky': 'sky',
                     'news': 'news',
                     'howe': 'howe',
                     'atlas': 'atlas',
                     'cosmos': 'cosmos',
                     'bias': 'bias',
                     'andes': 'andes'}

exceptional_early_exit_post_1a = ['inning', 'outing', 'canning', 'herring',
                                  'earring', 'proceed', 'exceed', 'succeed']


def stem(word):
    """The main entry point in the old version of the API."""
    raise DeprecationWarning('stem() is deprecated starting with v1.0.0')


def algorithms():
    """Get a list of the names of the available stemming algorithms.

    The only algorithm currently supported is the "english", or porter2,
    algorithm.
    """
    return ['english']


def version():
    """Get the version number of the stemming module.

    This is the version number of the Stemmer module as a whole (not for an
    individual algorithm).
    """
    return '1.0.0'


class PorterStemmer(Stemmer):
    """An instance of a stemming algorithm.

    When creating a Stemmer object, there is one required argument:
    the name of the algorithm to use in the new stemmer. A list of the
    valid algorithm names may be obtained by calling the algorithms()
    function in this module. In addition, the appropriate stemming algorithm
    for a given language may be obtained by using the 2 or 3 letter ISO 639
    language codes.
    """
    max_cache_size = 10000

    def __init__(self, algorithm='english', cache_size=None):
        if algorithm not in ['english', 'eng', 'en']:
            raise KeyError("Stemming algorithm '%s' not found" % algorithm)
        if cache_size:
            self.max_cache_size = cache_size

    def stem(self, sentence):
        """Stem a list of words.

        This takes a single argument, words, which must be a sequence,
        iterator, generator or similar.

        The entries in words should either be UTF-8 encoded strings,
        or a unicode objects.

        The result is a list of the stemmed forms of the words. If the word
        supplied was a unicode object, the stemmed form will be a unicode
        object: if the word supplied was a string, the stemmed form will
        be a UTF-8 encoded string.
        """
        # if sentence is a Sentence
        if isinstance(sentence, Sentence):
            if sentence.is_tokenized:
                if sentence.is_lemmatized:
                    warnings.warn("Sentence was stemmed, nothing to do")
                else:
                    stems = [PorterStemmer._stem(token.string)
                             for token in sentence]
                    sentence.append_tags(lemmas=stems)
                    sentence.pipeline.append(str(self))
            else:
                raise Exception("Sentence is not tokenized.")
        else:
            raise TypeError("Type of sentence not recognized")

    def batch_stem(self, sentences):
        if all([isinstance(sentence, Sentence) and sentence.is_tokenized
                for sentence in sentences]):
            for sentence in sentences:
                if sentence.is_lemmatized:
                    warnings.warn("Sentence was stemmed, nothing to do")
                else:
                    stems = [PorterStemmer._stem(token.string)
                             for token in sentence]
                    sentence.append_tags(lemmas=stems)
                    sentence.pipeline.append(str(self))

    @classmethod
    def _stem(cls, word):
        was_unicode = False

        if isinstance(word, unicode):
            was_unicode = True
            try:
                word = word.encode('ascii')
            except:
                return word

        if len(word) <= 2:
            return word
        word = remove_initial_apostrophe(word)

        # handle some exceptional forms
        if word in exceptional_forms:
            return exceptional_forms[word]

        word = capitalize_consonant_ys(word)
        r1 = get_r1(word)
        r2 = get_r2(word)
        word = step_0(word)
        word = step_1a(word)

        # handle some more exceptional forms
        if word in exceptional_early_exit_post_1a:
            return word

        word = step_1b(word, r1)
        word = step_1c(word)
        word = step_2(word, r1)
        word = step_3(word, r1, r2)
        word = step_4(word, r2)
        word = step_5(word, r1, r2)
        word = normalize_ys(word)

        if was_unicode:
            return word.decode('utf-8')
        return word


class CoreNLPLemmatizer(Stemmer):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.parser = CoreNLP(annotators=["tokenize", "ssplit", "pos", "lemma"],
                              ssplit_isOneSentence="true")

    def stem(self, sentence, build=False):
        # if sentence is a string/unicode
        if isinstance(sentence, basestring):
            # get only first (and only) sentence
            parsed_sentence = self.parser.parse(sentence)[0]
            if build:
                s = Sentence(string=sentence)
                for token in parsed_sentence.tokens:
                    s.append(string=token.word,
                             lemma=token.lemma)
                s.pipeline.append(str(self))
                return s
            else:
                return parsed_sentence

        # if sentence is a Sentence
        elif isinstance(sentence, Sentence):
            if sentence.is_lemmatized:
                warnings.warn("Sentence was stemmed, nothing to do")
            else:
                if sentence.is_tokenized:
                    string = " ".join(t.string for t in sentence)
                    if "tokenize_whitespace" not in self.kwargs:
                        warnings.warn("Adding tokenize_whitespace kwarg")
                        new_parser = CoreNLP(annotators=["tokenize", "ssplit", "pos", "lemma"],
                                             ssplit_isOneSentence="true",
                                             tokenize_whitespace="true",
                                             **self.kwargs)
                        # get only first (and only) sentence
                        parsed_sentence = new_parser.parse(string)[0]
                    else:
                        parsed_sentence = self.parser.parse(string)[0]
                    # process result
                    sentence.append_tags(lemmas=[t.lemma for t in parsed_sentence.tokens])
                else:
                    # get only first (and only) sentence
                    parsed_sentence = self.parser.parse(sentence.string)[0]
                    for token in parsed_sentence.tokens:
                        sentence.append(string=token.word,
                                        start=token.start,
                                        end=token.end,
                                        lemma=token.lemma)
                sentence.pipeline.append(str(self))
        else:
            raise TypeError("Type of sentence not recognized")

    def batch_stem(self, sentences, build=False):
        # if sentences are a strings/unicode
        if all(isinstance(s, basestring) for s in sentences):
            parsed_sentences = [result[0] for result in
                                self.parser.batch_parse(sentences)]
            if build:
                result = []
                for parsed_sentence in parsed_sentences:
                    s = Sentence(string=parsed_sentence.string)
                    for token in parsed_sentence.tokens:
                        s.append(string=token.word,
                                 lemma=token.lemma)
                    s.pipeline.append(str(self))
                    result.append(s)
                return result
            else:
                return parsed_sentences

        # if sentences are Sentence objects
        elif all(isinstance(s, Sentence) for s in sentences):
            # if all sentences are tokenized
            if all(s.is_tokenized for s in sentences):
                strings = [" ".join(t.string for t in s) for s in sentences]
                if "tokenize_whitespace" not in self.kwargs:
                    warnings.warn("Adding tokenize_whitespace kwarg")
                    new_parser = CoreNLP(annotators=["tokenize", "ssplit", "pos", "lemma"],
                                         ssplit_isOneSentence="true",
                                         tokenize_whitespace="true",
                                         **self.kwargs)
                    # get only first (and only) sentence
                    parsed_sentences = [result[0] for result in
                                        new_parser.batch_parse(strings)]
                else:
                    parsed_sentences = [result[0] for result in
                                        self.parser.batch_parse(strings)]
            else:
                strings = [sentence.string for sentence in sentences]
                parsed_sentences = [result[0] for result in
                                    self.parser.batch_parse(strings)]

            # process results
            for i, parsed_sentence in enumerate(parsed_sentences):
                if sentences[i].is_lemmatized:
                    warnings.warn("Sentence " + str(i) + " was stemmed, nothing to do")
                else:
                    sentences[i].append_tags(lemmas=[t.lemma for t in parsed_sentence.tokens])
                    sentences[i].pipeline.append(str(self))
        else:
            raise TypeError("Type of sentences not recognized")
