#!/usr/bin/python
# -*- coding: utf-8 -*-

import fim
from itertools import combinations, product
from collections import namedtuple
import Levenshtein
from scipy.cluster.hierarchy import linkage, fcluster
from .utils import StemKey
from ..stopwords import NLTKStopwords

_UID = 0


def _uid():
    global _UID
    _UID += 1
    return _UID

# ----------- PARAMETERS AS DEFINED BY PAPERS ---------------------------------

MIN_COMPACT_SUPPORT = 2   # as in "Mining Opinion Features in Customer Reviews"
ADJ_NEARBY_DISTANCE = 3       # not defined in the papers?
MAX_COMPACT_DISTANCE = 3  # as in "Mining Opinion Features in Customer Reviews"
MAX_WORDS = 3             # as in "Mining Opinion Features in Customer Reviews"
MIN_SUPPORT = 1           # as in "Mining and Summarizing Customer Reviews"
MIN_PSUPPORT = 3          # as in "Mining Opinion Features in Customer Reviews"


# ----------  MAIN CLASS  -----------------------------------------------------

class FreqAspectExtractor(object):

    def __init__(self,
                 stopwords=NLTKStopwords(),
                 min_support=MIN_SUPPORT,
                 max_words=MAX_WORDS,
                 min_psupport=MIN_PSUPPORT,
                 min_compact_support=MIN_COMPACT_SUPPORT,
                 max_compact_distance=MAX_COMPACT_DISTANCE,
                 adj_key=StemKey(),
                 adj_win_size=ADJ_NEARBY_DISTANCE ,
                 match=85,
                 compactness=True,
                 redundancy=True,
                 infrequent=True):
        """
        Model to extract aspects using the algorithm by Hu et al. (2004)

            stopwords             : iterable of strings to use as stopwords
            min_support           : int, minimum support of an item set
                                    (positive: percentage, negative: absolute
                                    number of transactions)
            min_compact_support   : int minimum number of compact sentences
                                    of an aspect
            max_words             : int, maximum number of word on each aspect,
            max_compact_distance  : int, maximum distance between consecutive
                                    words in an aspect
            adj_win_size          : int, maximum distance to look for
                                    adjectives near an aspect on a sentence
            min_psupport          : int, minimum pure support of an aspect
            adj_key               : lambda function to extract adjectives
            match                 : int, minimum similarity ratio (0-100] for
                                    matching (use <100 for fuzzy) default=
            compactness           : boolean, True to run "compactness pruning"
            redundancy            : boolean, True to run "redundancy pruning"
            infrequent            : boolean, True to also extract infrequent
                                    aspects
        """
        self.params = {"stopwords": stopwords,
                       "min_support": min_support,
                       "max_words": max_words,
                       "min_psupport": min_psupport,
                       "min_compact_support": min_compact_support,
                       "max_compact_distance": max_compact_distance,
                       "adj_key": adj_key,
                       "adj_win_size": adj_win_size,
                       "match": match,
                       "compactness": compactness,
                       "redundancy": redundancy,
                       "infrequent": infrequent}

    def extract(self, sentences):

        assert all(s.is_tagged for s in sentences)
        assert all(s.is_lemmatized for s in sentences)
        assert all(s.is_chunked for s in sentences)

        stopwords = self.params["stopwords"]
        min_support = self.params["min_support"]
        max_words = self.params["max_words"]
        min_psupport = self.params["min_psupport"]
        min_compact_support = self.params["min_compact_support"]
        max_compact_distance = self.params["max_compact_distance"]
        nearby_distance = self.params["adj_win_size"]
        adjective_key = self.params["adj_key"]
        match = self.params["match"]
        compactness = self.params["compactness"]
        redundancy = self.params["redundancy"]
        infrequent = self.params["infrequent"]

        global Aspect
        Aspect = type("Aspect", (Aspect,), {"max_compact_distance": max_compact_distance,
                                            "nearby_distance": nearby_distance,
                                            "min_compact_support": min_compact_support})

        transactions = _get_transactions(sentences, stopwords)

        transactions, word_clusters = _transactions_fuzzy_matching(transactions, match)
        candidates = _get_frequent_candidates(transactions, min_support, max_words)
        aspects = _get_aspects(candidates, sentences, Aspect, word_clusters=word_clusters)
        if compactness:
            _compactness_pruning(aspects)
        if redundancy:
            _redundancy_pruning(aspects, min_psupport)
        if infrequent:
            adjectives = _extract_nearby_adjectives(aspects, key=adjective_key)
            infreq_candidates = _get_infrequent_candidates(sentences, aspects, adjectives, stopwords, match)
            _get_aspects(infreq_candidates, sentences, Aspect, aspects=aspects)

        return aspects

# ----------  CLASSES  --------------------------------------------------------


class Aspect(object):

    min_compact_support = MIN_COMPACT_SUPPORT
    max_compact_distance = MAX_COMPACT_DISTANCE
    nearby_distance = ADJ_NEARBY_DISTANCE

    def __init__(self, stem_sets, aspects):
        self.aspects = aspects
        self.stem_sets = stem_sets
        self.ais = dict()
        self.id = _uid()

    def __str__(self, *args, **kwargs):
        return "Aspect<{0}>".format(self.id)

    def __repr__(self, *args, **kwargs):
        return self.__str__()

    def __len__(self):
        return len(list(self.stem_sets)[0])

    @property
    def compact_aspects(self):
        compact_aspects = set([])
        for sentence in self.compact_sentences:
            for compact_aspect in sentence.compact_aspects:
                real_compact_aspect = " ".join([token.string
                                                for token in compact_aspect])
                compact_aspects.add(real_compact_aspect)
        return compact_aspects

    @property
    def is_compound(self):
        if len(self) > 1:
            return True
        return False

    @property
    def sentences(self):
        return self.ais.values()

    @property
    def sentence_ids(self):
        return self.ais.keys()

    @property
    def has_super(self):
        for other_aspect in self.aspects:
            for this_stem_set in self.stem_sets:
                for other_stem_set in other_aspect.stem_sets:
                    if this_stem_set < other_stem_set:
                        return True
        return False

    def append(self, sentence):
        aspectinsentence = self.AspectInSentence(self, sentence)
        self.ais[sentence.id] = aspectinsentence

    def remove(self, sentence_id):
        del self.ais[sentence_id]

    def in_sentence(self, sentence_id):
        return self.ais[sentence_id]

    @property
    def support(self):
        return len(self.ais)

    @property
    def is_fuzzy(self):
        return True if len(self.stem_sets) > 1 else False

    @property
    def is_compact(self):
        if len(self.compact_sentences) >= self.min_compact_support:
            return True
        return False

    @property
    def compact_sentences(self):
        return [ais for ais in self.ais.values() if ais.is_compact]

    @property
    def psupport(self):
        """
        p-support (pure support): p-support of
        feature ftr is the number of sentences that
        ftr appears in as a noun or noun phrase, and these
        sentences must contain no feature phrase that is a
        superset of ftr.

        """
        where_itself = set()
        where_subset = set()
        for aspect in self.aspects:
            for this_stem_set in self.stem_sets:
                for other_stem_set in aspect.stem_sets:
                    if this_stem_set <= other_stem_set:
                        if this_stem_set == other_stem_set:
                            where_itself |= set(aspect.sentence_ids)
                        else:
                            where_subset |= set(aspect.sentence_ids)
        return len(where_itself.difference(where_subset))

    class AspectInSentence(object):

        AspectStem = namedtuple('AspectStem', ['stem', 'positions'],
                                verbose=False)

        def __init__(self, aspect, sentence):
            self.aspect = aspect
            self.sentence = sentence

        @property
        def stem_sets(self):
            return self.aspect.stem_sets

        def __str__(self, *args, **kwargs):
            return "Aspect<{0} in Sentence {1}>".format(self.aspect.id,
                                                        self.sentence.id)

        def __repr__(self, *args, **kwargs):
            return self.__str__()

        @property
        def is_compact(self):
            if self.aspect.is_compound:
                if any([len(compact_position)
                        for compact_position in self.compact_positions]):
                    return True
                return False
            return True

        @property
        def positions(self):
            poss = []
            for stem_set in self.stem_sets:
                pos = []
                for stem in stem_set:
                    stem_positions = [token.index for token in self.sentence
                                      if token.stem == stem]
                    pos.append(self.AspectStem(stem, stem_positions))
                poss.append(pos)
            return poss

        @property
        def compact_positions(self):
            result = []
            for fuzzy_aspect_position in self.positions:
                compact_positions = []
                for i, aspectstem in enumerate(fuzzy_aspect_position):
                    if not aspectstem.positions:
                        combinations = []
                        break
                    if i:
                        combinations = combine(combinations,
                                               aspectstem.positions,
                                               Aspect.max_compact_distance)
                    else:
                        combinations = [[stem_position]
                                        for stem_position
                                        in aspectstem.positions]
                if combinations:
                    compact_positions.append([sorted(combination)
                                              for combination in combinations
                                              if len(combination) == len(self.aspect)])
                if compact_positions:
                    result += compact_positions
            return result

        @property
        def compact_aspects(self):
            if self.is_compact:
                compact_aspects = []
                for fuzzy_compact in self.compact_positions:
                    if fuzzy_compact:
                        for compact_position in fuzzy_compact:
                            compact_aspects += [[self.sentence[index]
                                                 for index in compact_position]]
                return compact_aspects
            else:
                return []

        @property
        def nearby_adjectives(self):
            adjectives = []
            sent_len = len(self.sentence)
            for fuzzy_compact in self.compact_positions:
                if fuzzy_compact:
                    for position in fuzzy_compact:
                        start_pos_i = position[0] if position[0]-Aspect.nearby_distance < 0 else 0
                        start_pos_f = position[-1] + 1
                        end_pos_i = position[-1]
                        end_pos_f = position[-1] if position[-1]+Aspect.nearby_distance > sent_len else sent_len

                        for adjective in self.sentence.adjectives:
                            if adjective.index in range(start_pos_i,
                                                        start_pos_f):
                                adjectives.append(adjective)

                            if adjective.index in range(end_pos_i, end_pos_f):
                                adjectives.append(adjective)
            return adjectives

        @property
        def id(self): 
            return self.sentence.id

        @property
        def aspect_id(self):
            return self.aspect.id


# ----------  FUNCTIONS  ------------------------------------------------------

def distance(position_a, position_b):
    return abs(position_a - position_b)


def combine(combinations, stem_positions, max_compact_distance):
    """
    Generates combinations of stems sets in the rule that distance
    between all stems inside a set is up to max_compact_distance.
    Input:
        combinations:  existing combinations of stemsets positions
        stem_positions: position of a new stem that are wished to add to the
        given combination

        max_compact_distance: maximum distance between elements
        inside a stemset
    """
    new_combinations = []
    for combination in combinations:
        for stem_position in stem_positions:
            for position in combination:
                if distance(stem_position, position) <= max_compact_distance:
                    new_combination = combination[:] + [stem_position]
                    if new_combination not in new_combinations:
                        new_combinations.append(new_combination)
    return new_combinations


def adjective_fuzzy_matching(token, adjectives, match):
    """
    Given a token and a list of terms to match, returns True if
    the stem of the token matches any of the items in the list.
    Input:
        token: Token object to match
        adjectives: list of items to match the Token
        match: minimum ratio (0-100) for matching
    """
    for adjective in adjectives:
        if Levenshtein.ratio(str(token.stem), str(adjective)) >= match:
            return True
    return False


def _transactions_fuzzy_matching(transactions, match):
    """
    Runs fuzzy matching on the transactions, by applying a complete linkage
    hierarchical clustering algorithm to the set of different itemsets in the
    transactions. For clustering, the similarity ratio as given by
    fuzzywuzzy.ratio is used as the distance measure
    Input:
        transactions: list of tuples representing items on each transaction
        match: minimum similarity ratio (0 to 100) for clustering
    Output:
        transactions: new version of the transactions, where each item has been
                      replaced by the first item on its corresponding cluster
        word_clusters: dictionary that maps the cluster for each item
        in the transactions
    """
    words = set([])
    for transaction in transactions:
        words |= set(transaction)
    words = sorted(words)
    l = [((a, b), 100-Levenshtein.ratio(str(a), str(b)))
         for a, b in combinations(words, 2)]
    d = [value for pair, value in l]
    r = linkage(d, 'complete')
    clusters_index = fcluster(r, 100-match, "distance")
    clusters = {}
    for obs_i, cluster_i in enumerate(clusters_index):
        if cluster_i in clusters:
            clusters[cluster_i].append(words[obs_i])
        else:
            clusters[cluster_i] = [words[obs_i]]

    word_clusters = {word: clusters[clusters_index[i]]
                     for i, word in enumerate(words)}
    new_transactions = []
    for transaction in transactions:
        new_transaction = tuple(set(([word_clusters[word][0]
                                      for word in transaction])))
        new_transactions.append(new_transaction)
    return new_transactions, word_clusters


def _get_aspects(candidates, sentences, Aspect, word_clusters=None, aspects=None):
    """
    Creates a dict of id:Aspect from sentences, candidates (sets of stemsets)
    and their clusters When aspects are also provided, adds the
    new aspects to the dict. Given a frequent itemset of length 2 (a,b),
    the corresponding aspect is represented as the set of all possible
    combinations of stems a_i and b_j, where each a_i and b_j come from
    the terms in the cluster of a and b respectively.
    """
    returns = False
    if aspects is None:
        aspects = []
        returns = True

    if word_clusters:
        new_candidates = []
        for candidate, support in candidates:
            candidate_list = word_clusters[candidate[0]]
            if len(candidate) > 1:
                for i, word in enumerate(candidate[1:]):
                    if i:
                        candidate_list = [tuple(list(a)+[b])
                                          for a, b in product(candidate_list,
                                                              word_clusters[word])]
                    else:
                        candidate_list = [o for o in product(candidate_list,
                                                             word_clusters[word])]
            elif len(candidate) == 1:
                if len(candidate_list) == 1:
                    candidate_list = [tuple(candidate_list, )]
                else:
                    candidate_list = [(c, ) for c in candidate_list]
            new_candidates.append(frozenset([frozenset(c)
                                             for c in candidate_list]))
    else:
        new_candidates = candidates

    for stem_sets in new_candidates:
        aspect = Aspect(stem_sets, aspects)
        for sentence in sentences:
            sentence_stems = set(sentence.lemmata)
            for stem_set in stem_sets:
                if stem_set <= sentence_stems:
                    aspect.append(sentence)
        aspects.append(aspect)

    if returns:
        return aspects


def _get_transactions(sentences, stopwords):
    """
    Uses nouns and noun phrases in the sentences to generate
    transactions, as list of tuples, to be used for aspect extraction
    """
    transactions = []
    for sentence in sentences:
        sentence_transactions = set()
        # extract nouns
        for token in sentence.nouns:
            if len(token.string) > 1 and token.string.lower() not in stopwords:
                id = frozenset([token.stem])
                sentence_transactions |= id
        # extract phrase nouns
        for chunk in sentence.nps:
            id = frozenset([token.stem for token in chunk
                            if len(token.string) > 1 and token.string.lower() not in stopwords])
            sentence_transactions |= id

        transactions.append(tuple(sentence_transactions))
    return transactions


def _get_frequent_candidates(transactions, min_support, max_words):
    """
    Runs an frequent itemset mining algorithms on the transactions
        Input:
            transactions: list of tuples representing items on each transaction
            min_support: minimum support of an item set (positive: percentage,
                         negative: absolute number)
            max_words: maximum number of items in each frequent set
        Output:
            list of extracted frequent itemset in the form (itemsets, support)
    """
    return fim.fim(transactions, supp=min_support, zmax=max_words)


def _compactness_pruning(aspects):
    """
    Compactness pruning: This method checks features that
    contain at least two words and remove those that are likely
    to be meaningless.

    The idea of compactness pruning is to prune those candidate
    features whose words do not appear together.

    Athors use distances among the words in a candidate
    feature phrase (itemset) to do the pruning.

    Definition 1: compact phrase
      - Let f be a frequent feature phrase and f contains n
        words. Assume that a sentence s contains f and the
        sequence of the words in f that appear in s is: w1, w2,
        …, wn. If the word distance in s between any two
        adjacent words (wi and wi+1) in the above sequence is
        no greater than 3, then we say f is compact in s.

     - If f occurs in m sentences in the review database, and
       it is compact in at least 2 of the m sentences, then we
       call f a compact feature phrase.
    """
    for index, aspect in enumerate(aspects):
        if not aspect.is_compact:
            # print("removing " + str(aspect))
            aspects.pop(index)
        else:
            for sentence in aspect.sentences:
                if not sentence.is_compact:
                    aspect.remove(sentence.id)


def _redundancy_pruning(aspects, min_psupport):
    """
    Redundancy pruning: In this step, authors focus on removing
    redundant features that contain single words. To describe
    redundant features

    Authors use the minimum p-support to prune those redundant
    features. If a feature has a p-support lower than the
    minimum p-support (in our system, we set it to 3) and the
    feature is a subset of another feature phrase (which
    suggests that the feature alone may not be interesting).
    """
    for index, aspect in enumerate(aspects):
        if aspect.has_super and aspect.psupport <= min_psupport:
            # print("removing " + str(aspect))
            aspects.pop(index)


def _extract_nearby_adjectives(aspects, key=lambda x: x.stem):
    """Extracts adjectives appearing any aspect on all sentences"""
    adjectives = set()
    for aspect in aspects:
        for sentence in aspect.sentences:
            adjectives |= set([key(adj) for adj in sentence.nearby_adjectives])
    return adjectives


def _get_frequent_aspect_sentences(aspects, sentences):
    """Gets the list of sentences that contain any frequent aspect"""
    frequent_ids = set()
    for aspect in aspects:
        frequent_ids |= set(aspect.sentence_ids)
    return [sentences[id] for id in frequent_ids]


def _get_infrequent_candidates(sentences, aspects, adjectives, stopwords, match):
    """
    For each sentence in the review database, if it contains
    no frequent feature but one or more opinion words, find
    the nearest noun/noun phrase of the opinion word. The
    noun/noun phrase is then stored in the feature set as an
    infrequent feature.
    """
    sentences = {sentence.id: sentence for sentence in sentences}
    aspect_keys = [fuzzy_aspect for aspect in aspects
                   for fuzzy_aspect in aspect.stem_sets]

    # all sentence ids
    all_ids = set(sentences.keys())

    # ids of sentences with any frequent aspect
    frequent_ids = set()
    for aspect in aspects:
        frequent_ids |= set(aspect.sentence_ids)

    # ids of sentences without any frequent aspect
    no_aspect_ids = all_ids - frequent_ids

    candidates = []

    for sentence_id in no_aspect_ids:
        sentence = sentences[sentence_id]
        closest, min_dist = (None, float('Inf'))
        for token in sentence:
            if adjective_fuzzy_matching(token, adjectives, match):
                for noun in sentence.nouns:
                    if distance(noun.index, token.index) < min_dist:
                        closest = [noun]
                        min_dist = distance(noun.index, token.index)

                for np in sentence.nps:
                    np_min = min([distance(token.index, np.start),
                                  distance(token.index, np.stop)])
                    if np_min < min_dist:
                        closest = np
                        min_dist = np_min

        aspect = set()
        if closest:
            for term in closest:
                if term not in stopwords and term in sentence.nouns:
                    aspect.add(term.stem.lower())

        if aspect and aspect not in aspect_keys:
            candidate = frozenset(aspect)
            candidates.append(frozenset([candidate]))
            aspect_keys.append(candidate)

    return candidates
