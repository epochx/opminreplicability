#!/usr/bin/python
# -*- coding: utf-8 -*-

import warnings

from ..rep import Sentence
from ..corenlp import CoreNLP
from ..senna import Senna


class ConstParser():

    def __init__(self):
        raise NotImplemented

    def __str__(self):
        return self.__class__.__name__

    def parse(self):
        raise NotImplementedError

    def batch_parse(self):
        raise NotImplementedError


class CoreNLPConstParser(ConstParser):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.kwargs["ssplit_isOneSentence"] = "True"
        self.parser = CoreNLP(annotators=["tokenize", "ssplit", "pos", "lemma", "parse"],
                              **self.kwargs)

    def parse(self, sentence, build=False):
        # if sentence is a String/Unicode
        if isinstance(sentence, basestring):
            # get only first document
            parsed_sentence = self.parser.parse(sentence)[0]
            if build:
                return self._process_sentence(sentence, parsed_sentence)
            else:
                return parsed_sentence

        # if sentence is a Sentence object
        elif isinstance(sentence, Sentence):
            # if sentence is tokenized
            if sentence.is_tokenized:
                string = " ".join(t.string for t in sentence)
                if "tokenize_whitespace" not in self.kwargs:
                    warnings.warn("Adding tokenize_whitespace kwarg")
                    new_parser = CoreNLP(annotators=["tokenize", "ssplit", "pos", "lemma", "parse"],
                                         tokenize_whitespace="true", **self.kwargs)
                    # get only first document for each sentence
                    parsed_sentence = new_parser.parse(string)[0]
                else:
                    # get only first document for each sentence
                    parsed_sentence = self.parser.parse(string)[0]
            else:
                # get only first document
                parsed_sentence = self.parser.parse(sentence.string)[0]
            # process result
            self._process_sentence(sentence, parsed_sentence)
        else:
            raise TypeError("Sentence type nor supported")

    def batch_parse(self, sentences, build=False):
        # if all sentences are strings/unicode
        if all(isinstance(s, basestring) for s in sentences):
            parsed_sentences = [result[0] for result in
                                self.parser.batch_parse(sentences)]
            if build:
                return [self._process_sentence(sentences[i], parsed_sentence)
                        for i, parsed_sentence in enumerate(parsed_sentences)]
            else:
                return parsed_sentences

        # if all sentences are Sentence objects
        elif all(isinstance(s, Sentence) for s in sentences):
            # if all sentences are tokenized
            if all([s.is_tokenized for s in sentences]):
                strings = [" ".join([t.string for t in sentence])
                           for sentence in sentences]
                if "tokenize_whitespace" not in self.kwargs:
                    warnings.warn("Adding tokenize_whitespace kwarg")
                    new_parser = CoreNLP(annotators=["tokenize", "ssplit", "pos", "lemma", "parse"],
                                         tokenize_whitespace="true", **self.kwargs)
                    # get only first document for each sentence
                    parsed_sentences = [result[0] for result in
                                        new_parser.batch_parse(strings)]
                else:
                    # get only first document for each sentence
                    parsed_sentences = [result[0] for result in
                                        self.parser.batch_parse(strings)]
            else:
                strings = [sentence.string for sentence in sentences]
                # get only first document for each sentence
                parsed_sentences = [result[0] for result in
                                    self.parser.batch_parse(strings)]
            # process results
            for i, parsed_sentence in enumerate(parsed_sentences):
                self._process_sentence(sentences[i], parsed_sentence)
        else:
            raise TypeError("Sentence type nor supported")

    def _process_sentence(self, sentence, parsed_sentence):
        """
        Process a raw Stanford parsed sentence (parsed_sentence)
        and add tokens to sentence.
        """
        should_return = False
        if isinstance(sentence, basestring):
            sentence = Sentence(string=sentence)
            should_return = True

        if not sentence.is_tokenized:
            for token in parsed_sentence.tokens:
                sentence.append(string=token.word,
                                start=token.start,
                                end=token.end,
                                lemma=token.lemma,
                                pos_tag=token.POS)
        else:
            if not sentence.is_tagged:
                tags = [t.POS for t in parsed_sentence.tokens]
                sentence.append_tags(pos_tags=tags)

            if not sentence.is_lemmatized:
                lemmas = [t.lemma for t in parsed_sentence.tokens]
                sentence.append_tags(lemmas=lemmas)

        # eliminate the ROOT node
        syntax_tree = parsed_sentence.syntax_tree

        if "(ROOT " in syntax_tree:
            syntax_tree = syntax_tree.strip().replace("(ROOT ", "")[:-1]

        sentence.tree = syntax_tree
        sentence.pipeline.append(str(self))

        if should_return:
            return sentence


class SennaConstParser(ConstParser):

    def __init__(self, args=[]):
        self.parser = Senna()
        self.args = ["pos", "chk", "psg", "iobtags"] + args

    def parse(self, sentence, build=False):
        # if sentence is a string/unicode
        if isinstance(sentence, basestring):
            parsed_sentence = self.parser.parse(sentence, args=self.args)
            if build:
                return self._process_sentence(sentence, parsed_sentence)
            else:
                return parsed_sentence

        # if sentence is a Sentence object
        elif isinstance(sentence, Sentence):
            # if sentence is tokenized
            if sentence.is_tokenized:
                string = " ".join(token.string for token in sentence)
                if "usrtokens" not in self.args:
                    warnings.warn("Adding the usrtokens arg")
                    args = self.args + ["usrtokens"]
                else:
                    args = self.args
                parsed_sentence = self.parser.parse(string, args=args)
            else:
                string = sentence.string
                parsed_sentence = self.parser.parse(string, args=self.args)
            # process result
            self._process_sentence(sentence, parsed_sentence)
        else:
            raise TypeError("Sentence type nor supported")

    def batch_parse(self, sentences, build=False):
        # if all sentences are strings/unicodes
        if all(isinstance(s, basestring) for s in sentences):
            parsed_sentences = self.parser.batch_parse(sentences)
            if build:
                return [self._process_sentence(sentences[i], parsed_sentence)
                        for i, parsed_sentence in enumerate(parsed_sentences)]
            else:
                return parsed_sentences

        # if all sentences are Sentence objects
        elif all(isinstance(s, Sentence) for s in sentences):
            if all([s.is_tokenized for s in sentences]):
                strings = [" ".join([t.string for t in sentence])
                           for sentence in sentences]
                if "usrtokens" not in self.args:
                    warnings.warn("Adding the usrtokens arg")
                    args = self.args + ["usrtokens"]
                else:
                    args = self.args
                parsed_sentences = self.parser.batch_parse(strings, args=args)
            else:
                strings = [sentence.string for sentence in sentences]
                parsed_sentences = self.parser.batch_parse(strings)

            # process results
            for i, parsed_sentence in enumerate(parsed_sentences):
                self._process_sentence(sentences[i], parsed_sentence)
        else:
            raise TypeError("Sentence type nor supported")

    def _process_sentence(self, sentence, parsed_sentence):
        """
        Process a raw Stanford parsed sentence (parsed_sentence)
        and add tokens to sentence.
        """
        should_return = False
        if isinstance(sentence, basestring):
            sentence = Sentence(string=sentence)
            should_return = True

        if not sentence.is_tokenized:
            for token in parsed_sentence.tokens:
                sentence.append(string=token.word,
                                start=token.start,
                                end=token.end,
                                pos_tag=token.pos,
                                iob_tag=token.chunk)
        else:
            if not sentence.is_tagged:
                tags = [t.pos for t in parsed_sentence.tokens]
                sentence.append_tags(pos_tags=tags)

            if not sentence.is_chunked:
                iob_tags = [t.chunk for t in parsed_sentence.tokens]
                sentence.append_tags(iob_tags=iob_tags)

        syntax_tree = parsed_sentence.syntax_tree
        if "(S1" in syntax_tree:
            syntax_tree = syntax_tree.replace("(S1", "")[:-1]

        sentence.tree = syntax_tree
        sentence.pipeline.append(str(self))

        if should_return:
            return sentence
