#!/usr/bin/python
# -*- coding: utf-8 -*-

import warnings

from ..rep import Sentence
from ..corenlp import CoreNLP


class DepParser(object):

    def __init__(self):
        raise NotImplemented

    def parse(self):
        raise NotImplemented

    def batch_parse(self):
        raise NotImplemented

    def __str__(self):
        return self.__class__.__name__


class CoreNLPDepParser(DepParser):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.kwargs["ssplit_isOneSentence"] = "True"
        self.parser = CoreNLP(annotators=["tokenize", "ssplit", "pos", 
                                          "lemma", "parse", "depparse"],
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
                    new_parser = CoreNLP(annotators=["tokenize", "ssplit", "pos",
                                                     "lemma", "parse", "depparse"],
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
                    new_parser = CoreNLP(annotators=["tokenize", "ssplit", "pos",
                                                     "lemma", "parse", "depparse"],
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

        deps = []

        for dep in parsed_sentence.dependencies:
            head_index = int(dep.head.index)-1
            dep_index = int(dep.dependent.index)-1
            deps.append((head_index, dep.label, dep_index))

        if deps:
            sentence.append_tags(rels=deps)

        # eliminate the ROOT node
        syntax_tree = parsed_sentence.syntax_tree

        if "(ROOT " in syntax_tree:
            syntax_tree = syntax_tree.strip().replace("(ROOT ", "")[:-1]

        sentence.tree = syntax_tree
        sentence.pipeline.append(str(self))

        if should_return:
            return sentence

    def tree2deps(self, sentence, **options):
        """
        :param sentence:
        :param options:
        :return:
        """
        if isinstance(sentence, Sentence) \
        and sentence.is_parsed:
            tree = sentence._tree_string
            dependencies = self.parser.tree2deps(tree, **options)
            deps = []
            for dep in dependencies:
                head_index = int(dep.head.index)-1
                dep_index = int(dep.dependent.index)-1
                deps.append((head_index, dep.label, dep_index))

            if deps:
                sentence.append_tags(rels=deps)
                sentence.pipeline.append(str(self))
        else:
            raise Exception("Sentence is not parsed")

    def batch_tree2deps(self, sentences, **options):
        if all([isinstance(sentence, Sentence) and sentence.is_parsed
                for sentence in sentences]):
            trees = [s._tree_string for s in sentences]
            results = self.parser.batch_tree2deps(trees, **options)
            for i, dependencies in enumerate(results):
                deps = []
                for dep in dependencies:
                    head_index = int(dep.head.index) - 1
                    dep_index = int(dep.dependent.index) - 1
                    deps.append((head_index, dep.label, dep_index))
                if deps:
                    sentences[i].append_tags(rels=deps)
                    sentences[i].pipeline.append(str(self))
        else:
            raise Exception("A sentence is not parsed")





