#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from enlp.corpus.liu import CreativeMP3Player, ApexDVDPlayer, NikonCamera, CanonCamera, NokiaCellphone
from enlp.pipe.chunkers import CoNLL2000Chunker
from enlp.pipe.cparsers import SennaConstParser
from enlp.pipe.dparsers import CoreNLPDepParser
from enlp.pipe.stemmers import PorterStemmer

if __name__ == "__main__":

    Corpora = {"CreativeMP3Player": CreativeMP3Player,
               "ApexDVDPlayer": ApexDVDPlayer,
               "NikonCamera": NikonCamera,
               "CanonCamera": CanonCamera,
               "NokiaCellphone": NokiaCellphone}

    desc = "Help for process_datasets, a script that annotates a list of given corpora"

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--corpus', "-c",
                        nargs='*',
                        choices=Corpora,
                        help="Names of corpus to use. Allowed values are " + ', '.join(Corpora),
                        metavar='')

    annotators = ["Senna", "CoreNLP"]
    parser.add_argument('--annotators', "-a",
                        nargs='*',
                        choices=annotators,
                        help="Annotators to use to pre-process corpora. Allowed values are " + ', '.join(annotators),
                        metavar='')

    args = parser.parse_args()

    if args.corpus:
        corpus_names = args.corpus
    else:
        corpus_names = Corpora.keys()

    if args.annotators:
        annotator_names = args.annotators
    else:
        annotator_names = annotators

    for annotator_name in annotator_names:
        for corpus_name in corpus_names:
            Corpus = Corpora[corpus_name]
            corpus = Corpus()
            print "processing " + corpus.name
            if annotator_name == "CoreNLP":
                dparser = CoreNLPDepParser()
                chunker = CoNLL2000Chunker()

                dparser.batch_parse(corpus.sentences)
                chunker.batch_parse(corpus.sentences)

            if annotator_name == "Senna":
                cparser = SennaConstParser()
                dparser = CoreNLPDepParser()
                stemmer = PorterStemmer()
                cparser.batch_parse(corpus.sentences)
                stemmer.batch_stem(corpus.sentences)
                dparser.batch_parse(corpus.sentences)
            corpus.freeze()
