#!/usr/bin/python
# -*- coding: utf-8 -*-

from os import path
from collections import OrderedDict
import re

from ..settings import CORPORA_PATH
from ..rep import Sentence, Document, Aspect
from .utils import Corpus, CorpusError

OPINION_PATH = path.join(CORPORA_PATH, 'opinion')
LIU_CORPORA_PATH = path.join(OPINION_PATH, 'liu')

# --- LIU CORPUS CLASSES ------------------------------------------------------


class LiuCorpus(Corpus):
    """
    Class to read Liu corpus. Sentences in the corpus are extracted as
    the sentences attribute. Each sentence has an id, review_id and the
    aspects as a dict. Each aspect is also a dict including its type (u/n)
    and orientation (1 or -1). Id starts on 0 for each review, denoting
    the title
    """

    filepath = ""

    def __init__(self):
        """
        Representation of a line inside a corpus. CorpusRead Methods
        are used by this object to process each raw line as read from
        the source file
        """
        # self.name = filepath.split('/')[-1].replace('.txt', '')
        if self._check():
            self._read()
        else:
            raise CorpusError("Corpus was not properly built. Check for consistency")

    def __repr__(self):
        return "<LiuCorpus {0}>".format(self.name)

    @property
    def aspects(self):
        return self._aspects.values()

    @property
    def sentences(self):
        return self._sentences.values()

    @property
    def reviews(self):
        return self._reviews.values()

    def _check(self):
        counter = 1
        mfile = open(self.filepath, "r")
        for i in mfile.readlines():
            linea = i.replace('\n', '')
            if (self._is_liu_comment(linea) is False):
                if (self._is_new_comment(linea) is False):
                    partes = linea.split('##')
                    if (len(partes) < 2):
                        print counter
                        return False
                    tags = partes[0]
                    real_sentence = partes[1].replace("\r", "").replace("\n", "").strip()
                    if (real_sentence == ""):
                        print counter
                        return False
                    if (tags != ""):
                        aspects_list = tags.strip().split(",")
                        for aspect_item in aspects_list:
                            aspect = self._extract_aspect(aspect_item)
                            if (aspect == ""):
                                print counter
                                return False
                            orientation = self._extract_orientation(aspect_item)
                            if (orientation is None):
                                print counter
                                return False
            counter += 1
        return True

    def _read(self):
        self._comment_counter = 0
        self._sentence_counter = 1
        self._aspects = OrderedDict()
        self._reviews = OrderedDict()
        self._sentences = OrderedDict()

        for line in open(self.filepath, "r").readlines():
            self.parse_line(line)

    def parse_line(self, string):
        if not self._is_new_comment(string):
            aspects_string, string = string.split('##')
            string = string.replace("\r", "").replace('\n', '').strip()
            id = self._sentence_counter
            review_id = self._comment_counter

            review = self._reviews.get(review_id, None)
            if not review:
                review = Document(id=review_id)
                self._reviews[review_id] = review
            sentence = Sentence(string=string, id=id, document=review)
            sentence.aspects = []
            self._sentences[id] = sentence
            review.append(sentence) 
            if aspects_string:
                for aspect_string in aspects_string.strip().split(", "):
                    term = self._extract_aspect(aspect_string)
                    orientation = self._extract_orientation(aspect_string)
                    type = self._extract_type(aspect_string)
                    aspect = self._aspects.get(term, None)
                    if aspect:
                        aspect.append(sentence, orientation, type)
                    else:
                        self._aspects[term] = Aspect(term, sentence, orientation, type)
            self._sentence_counter += 1
        else:
            self._comment_counter += 1

    def _is_liu_comment(self, string):
        if (string == ("\n") or string == ("\r")):
            return True
        else:
            pos = re.search("\A\*+", string)
            if (pos is None):
                return False
            else:
                return True

    def _is_new_comment(self, string):
        pos = string.find("[t]")
        if (pos != -1):
            return True
        else:
            return False

    def _extract_aspect(self, string):
        pos = string.find("[")
        if (pos != -1):
            return string[:(pos)].strip()
        else:
            return ""

    def _extract_orientation(self, string):
        pos = string.find("+")
        if (pos != -1):
            return 1
        else:
            pos = string.find("-")
            if (pos != -1):
                return -1

    def _extract_type(self, string):
        if "[u]" in string:
            return "u"
        elif "[p]" in string:
            return "p"
        else:
            return "n"

CreativeMP3Player = type("CreativeMP3Player",
                         (LiuCorpus,),
                         {"filepath": path.join(LIU_CORPORA_PATH,
                                                'creative_mp3_player.txt')})

ApexDVDPlayer = type("ApexDVDPlayer",
                     (LiuCorpus,),
                     {"filepath": path.join(LIU_CORPORA_PATH,
                                            'apex_dvd_player.txt')})

NikonCamera = type("NikonCamera",
                   (LiuCorpus,),
                   {"filepath": path.join(LIU_CORPORA_PATH,
                                          'nikon_camera.txt')})

CanonCamera = type("CanonCamera",
                   (LiuCorpus,),
                   {"filepath": path.join(LIU_CORPORA_PATH,
                                          'canon_camera.txt')})

NokiaCellphone = type("NokiaCellphone",
                       (LiuCorpus,),
                       {"filepath": path.join(LIU_CORPORA_PATH,
                                              'nokia_cellphone.txt')})

__all__ = [CreativeMP3Player,
           ApexDVDPlayer,
           NikonCamera,
           CanonCamera,
           NokiaCellphone]
