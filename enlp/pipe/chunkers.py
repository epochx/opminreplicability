#!/usr/bin/python
# -*- coding: utf-8 -*-

import subprocess
import re

from ..settings import CHUNKLINK_PATH
from ..rep import Sentence
from ..senna import Senna


class Chunker():

    def __str__(self):
        return self.__class__.__name__

    def parse(self):
        raise NotImplementedError

    def batch_parse(self):
        raise NotImplementedError


class CoNLL2000Chunker(Chunker):

    def __init__(self, chunklink_path=CHUNKLINK_PATH):
        self.chunklink_path = chunklink_path

    def parse(self, sentence):
        if isinstance(sentence, Sentence) \
        and sentence.is_parsed:
            iob_tags = self.tree_to_chunks(sentence._tree_string)
            sentence.append_tags(iob_tags=iob_tags)
            sentence.pipeline.append(str(self))
        else:
            raise Exception("Sentences no supported")

    def batch_parse(self, sentences):
        if all([isinstance(sentence, Sentence) and sentence.is_parsed
                for sentence in sentences]):
            for sentence in sentences:
                iob_tags = self.tree_to_chunks(sentence._tree_string)
                sentence.append_tags(iob_tags=iob_tags)
                sentence.pipeline.append(str(self))
        else:
            raise Exception("A sentence is not supported")

    def tree_to_chunks(self, conll_parse_tree):
        """
        A wrapper around a modified version of the chunklink.pl file, used
        for the CONLL shared task 2000, which reads Penn treebank parses
        from STDIN and returns flat chunks

        Source: https://github.com/mgormley/concrete-chunklink
        """

        # Convert concrete Parse to a PTB style parse string
        # to use as stdin for chunklink.
        ptb_str = '( ' + conll_parse_tree + ' )\n'
        ptb_str = ptb_str.encode('ascii', 'replace')
        # ptb_str = ptb_str.encode('ascii', 'ignore')

        # Run the chunklink script and capture the output.
        try:
            # We expect the chunklink script to be a modified version 
            # which can read a tree from stdin.
            p = subprocess.Popen(['perl', self.chunklink_path],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 stdin=subprocess.PIPE)
            stdouterr = p.communicate(input=ptb_str)
            chunk_str = stdouterr[0]
            chunk_err = stdouterr[1]
            # print("Chunklink stdout:\n" + chunk_str) 
            # print("Chunklink stderr:\n" + chunk_err)
            chunk_tags = self.get_chunks(chunk_str)
            return chunk_tags

        except subprocess.CalledProcessError as e:
            # logging.error("Chunklink failed on tree: %s" % (ptb_str))
            raise e

    @staticmethod
    def get_chunks(chunk_str):
        '''Gets the column of B-I-O tags denoting the chunks
        from the output of the chunklink script.'''
        whitespace = re.compile(r"\s+")
        chunks = []
        lines = chunk_str.split("\n")
        for line in lines:
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            columns = whitespace.split(line)
            chunks.append(columns[3])
        return chunks


class SennaChunker(Chunker):

    def __init__(self, args=[]):
        self.parser = Senna()
        self.args = ["pos", "offsettags", "chk", "iobtags"] + args

    def parse(self, sentence, build=False):
        if isinstance(sentence, basestring):
            # get only first document
            parsed_sentence = self.parser.parse(sentence,
                                                args=self.args)
            if build:
                return self._process_sentence(sentence, parsed_sentence)
            else:
                return parsed_sentence

        elif isinstance(sentence, Sentence):
            if sentence.is_tokenized:
                if "usrtokens" not in self.args:
                    raise Exception("Add usrtokens to args please")
                else:
                    string = " ".join(token.string for token in sentence)
                # get only first document
            else:
                string = sentence.string
            parsed_sentence = self.parser.parse(string, args=self.args)
            self._process_sentence(sentence, parsed_sentence)
        else:
            raise TypeError("Sentence type nor supported")

    def batch_parse(self, sentences, build=False):
        if isinstance(sentences[0], basestring):
            parsed_sentences = self.parser.batch_parse(sentences,
                                                       args=self.args)
            if build:
                return [self._process_sentence(sentences[i], parsed_sentence)
                        for i, parsed_sentence in enumerate(parsed_sentences)]
            else:
                return parsed_sentences

        elif isinstance(sentences[0], Sentence):
            if any([s.is_tokenized for s in sentences]):
                if "usrtokens" not in self.args:
                    raise Exception("Add usrtokens to args please")
                else:
                    strings = [" ".join([t.string for t in sentence])
                               for sentence in sentences]
            else:
                strings = [sentence.string for sentence in sentences]
            # get only first document for each sentence
            parsed_sentences = self.parser.batch_parse(strings,
                                                       args=self.args)
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
                sentence.append(pos_tag=token.pos,
                                string=token.word,
                                start=token.start,
                                end=token.end,
                                iob_tag=token.chunk)
            sentence.pipeline.append(str(self))

        elif not sentence.is_tagged:
            pos_tags = [t.pos for t in parsed_sentence.tokens]
            sentence.append_tags(pos_tags=pos_tags)
            sentence.pipeline.append(str(self))

        elif not sentence.is_chunked:
            iob_tags = [t.chunk for t in parsed_sentence.tokens]
            sentence.append_tags(iob_tags=iob_tags)
            sentence.pipeline.append(str(self))

        if should_return:
            return sentence

