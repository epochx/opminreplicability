#!/usr/bin/python
# -*- coding: utf-8 -*-

from os import path

CODE_ROOT = path.dirname(path.realpath(__file__))
print "Using " + str(CODE_ROOT)

# Environment PATH settings
DATA_PATH = "path to data"
CORENLP_PATH = "path_to_stanford-corenlp-full-2015-12-09"
JAVA_HOME = "/usr/lib/jvm/java-8-openjdk-amd64"
SENNA_PATH = "path_to_senna"

CORPORA_PATH = path.join(DATA_PATH, "corpora")
CHUNKLINK_PATH = path.join(CODE_ROOT, "script/mod_chunklink_2-2-2000_for_conll.pl")
CONLLEVAL_PATH = path.join(CODE_ROOT, "script/conlleval.pl")
PICKLE_PATH = path.join(DATA_PATH, "pickle")
