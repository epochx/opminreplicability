#!/usr/bin/python
# -*- coding: utf-8 -*-

# --- CLASSES -----------------------------------------------------------------


class Aspect(object):

    def __init__(self, string, sentence, orientation, type, position=None):
        self.string = string
        self.airs = dict()
        self.airs[sentence.review.id] = AspectInReview(self, sentence,
                                                       orientation,
                                                       type, position)

    @property
    def aiss(self):
        # dict of term in sentence Objects
        all_items = [(key, value) for review in self.reviews
                     for (key, value) in review.aiss.items()]
        return {key: value for (key, value) in all_items}

    @property
    def reviews(self):
        return self.airs.values()

    @property
    def sentences(self):
        # dict of term in sentence Objects
        return [sentence for review in self.reviews
                for sentence in review.sentences]

    @property
    def type(self):
        if any([s.type == "n" for s in self.sentences]):
            return "n"
        else:
            return "u"

    @property
    def sentence_ids(self):
        return self.aiss.values()

    @property
    def review_ids(self):
        return self.airs.values()

    @property
    def support(self):
        return len(self.aiss)

    def in_sentence(self, sentence_id):
        return self.aiss[sentence_id]

    def in_review(self, review_id):
        return self.airs[review_id]

    def append(self, sentence, orientation, type, position=None):
        air = self.airs.get(sentence.review.id, None)
        if air:
            ais = air.append(sentence, orientation, type, position)
        else:
            air = AspectInReview(self, sentence, orientation, type, position)
            self.airs[sentence.review.id] = air


class AspectInReview(object):

    def __init__(self, aspect, sentence, orientation, type, position):
        self.aspect = aspect
        self.review = sentence.review

        # dict of term in sentence Objects
        self.aiss = dict()
        self.aiss[sentence.id] = AspectInSentence(self, sentence, orientation,
                                                  type, position)

    def __str__(self, *args, **kwargs):
        return "Aspect<{0} in Review {1}>".format(self.string, self.id)

    def __repr__(self, *args, **kwargs):
        return self.__str__()

    @property
    def sentences(self):
        return self.aiss.values()

    @property
    def string(self):
        return self.review.string

    @property
    def id(self):
        return self.review.id

    @property
    def support(self):
        return len(self.aiss)

    def append(self, sentence, orientation, type, position):
        ais = self.aiss.get(sentence.id, None)
        if ais:
            ais.append(position)
        else:
            self.aiss[sentence.id] = AspectInSentence(self, sentence,
                                                      orientation,
                                                      type, position)


class AspectInSentence(object):

    def __init__(self, aspect_in_review, sentence, orientation,
                 type, position):
        self.sentence = sentence
        self.orientation = orientation
        self.type = type
        self.positions = [position]
        self.air = aspect_in_review
        sentence.aspects.append(self)

    def __str__(self, *args, **kwargs):
        return "Aspect<{0} in Sentence {1}>".format(self.string, self.id)

    def __repr__(self, *args, **kwargs):
        return self.__str__()

    def append(self, position):
        if position not in self.positions:
            self.positions.append(position)

    @property
    def review(self):
        return self.air

    @property
    def aspect(self):
        return self.air.aspect

    @property
    def string(self):
        return self.aspect.string

    @property
    def id(self):
        return self.sentence.id
