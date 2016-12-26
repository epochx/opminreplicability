#!/usr/bin/python
# -*- coding: utf-8 -*-

# progress bar
# from ipywidgets import FloatProgress
# from IPython.display import display

import sys
import cPickle as pickle
from os import path, listdir, remove

# unpickle speedup trick
import gc

from enlp.settings import PICKLE_PATH

sys.setrecursionlimit(10000000)


class CorpusError(Exception):
    """
    Base Class for Corpus
    """
    pass


class Corpus(object):

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        return str(self)

    @property
    def pipeline(self):
        pipeline = self.sentences[0].pipeline
        assert all(sentence.pipeline == pipeline
                   for sentence in self.sentences)
        return pipeline

    def freeze(self):
        """
        Save a processed corpus.
        """
        if self.pipeline:
            try:
                filename = ".".join(self.pipeline + [self.name])
                filename = "{0}.pickle".format(filename)
                f = open(path.join(PICKLE_PATH, filename), "wb")
                pickle.dump(self, f)
                f.close()
            except Exception as e:
                if f:
                    f.close()
                remove(path.join(PICKLE_PATH, filename))
                print e

    @classmethod
    def list_frozen(cls):
        return [filename.replace("." + cls.__name__ + ".pickle", "").split(".")
                for filename in listdir(PICKLE_PATH)
                if cls.__name__ in filename]

    @classmethod
    def unfreeze(cls, pipeline):
        """
        TO DO
        """
        try:
            filename = ".".join(pipeline + [cls.__name__])
            f = open(path.join(PICKLE_PATH, "{0}.pickle".format(filename)), "rb")
            gc.disable()
            corpus = pickle.load(f)
            gc.enable()
            return corpus
        except Exception as e:
            print e
            return None