#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import OrderedDict, defaultdict
from .utils import LowerStringKey
from ..lexicon.manual import ManualLexicon
import Levenshtein
from fim import fim

_UID = 0


def _uid():
    global _UID
    _UID += 1
    return _UID

# ------- DEFINITIONS----------------------------------------------------------

MODS = frozenset(['mod', 'pnmod'])
OBJS = frozenset(['subj', 's', 'obj'])
MR = frozenset(['obj2', 'desc', 'mod', 'pnmod', 'subj', 's', 'obj'])
CONJ = frozenset(['conj'])
JJ = frozenset(['JJ', 'JJR', 'JJS'])
NN = frozenset(['NN', 'NNS'])
# NN = frozenset(['NN', 'NNS', 'NNP'])

# adding "n't" and "'t" for tokenization compatibility
NEGATIONS = ['not', "n't", "'t", 'however', 'but', 'despite', 'though',
             'except', 'although', 'oddly', 'aside']
CONJUNCTIONS = ['and', 'or']
PRODUCT_PHRASES = [['compare', 'to'], ['compare', 'with'],
                   ['compare', None, 'to'], ['compare', None, None, 'to'],
                   ['compare', None, 'with'], ['compare', None, None, 'with'],
                   ['better', 'than'], ['worse', 'than']]
DEALER_PHRASES = [['shop', 'with'], ['buy', 'from']]
NEGATION_LIMIT = 5
Q = 2
K = 1

BASIC_SEEDS = ManualLexicon('Base Seeds')
BASIC_SEEDS["good"] = 1.0
BASIC_SEEDS["bad"] = -1.0

# ------- MAIN CLASS ----------------------------------------------------------


class DoublePropagation(object):

    def __init__(self,
                 seeds=BASIC_SEEDS,
                 negations=NEGATIONS,
                 negation_limit=NEGATION_LIMIT,
                 conjunctions=CONJUNCTIONS,
                 product_phrases=PRODUCT_PHRASES,
                 dealer_phrases=DEALER_PHRASES,
                 Q=Q,
                 K=K,
                 key=LowerStringKey(),
                 match=90):
        """
        Input:
            - seeds            : dict of seed opinion words to use, (each is
                                 an object with .string, .orientation and
                                 .stem attributes)
                                 default=liu_lexicon
            - negations        : list of strings, words that convey negation
                                 default=NEGATIONS
            - negation_limit   : int, maximum distance to look for negations
                                 default=NEGATION_LIMIT
            - conjunctions     : list of strings, conjunctions to check aspects
                                 in a clause, default=CONJUNCTIONS 
            - product_phrases  : lisf of lists of strings, used in product
                                 phrase pruning, default=PRODUCT_PHRASES
            - dealer_phrases   : DEALER_PHRASES
            - Q                : int, Q parameter
            - K                : int, K parameter
            - key              : function, to apply to each word before
                                 extraction, LowerStringKey=lambda x: x.string.lower
            - match            : int (0-100), percentage of similarity for matching
                                 terms (use 100 for exact), default=90
        Output:
            tuple: (list aspects, list opinion_words)
        """

        self.params = {"seeds": seeds,
                       "negations": negations,
                       "negation_limit": negation_limit,
                       "conjunctions": conjunctions,
                       "product_phrases": product_phrases,
                       "dealer_phrases": dealer_phrases,
                       "Q": Q,
                       "K": K,
                       "key": key,
                       "match": match}

        if match < 100:
            self.matching = ExactMatching(key=key)
        else:
            self.matching = FuzzyMatching(match=match, key=key)

    def extract(self, sentences):
        """
        Returns the extracted aspects and opinion words.
        Input:
            - sentences: list of sentences
        Output:
            - tuple(list aspects, list opinion_words)
        """
        seeds = self.params["seeds"]
        negations = self.params["negations"]
        negation_limit = self.params["negation_limit"]
        conjunctions = self.params["conjunctions"]
        product_phrases = self.params["product_phrases"]
        dealer_phrases = self.params["dealer_phrases"]
        Q = self.params["Q"]
        K = self.params["K"]
        key = self.params["key"]
        matching = self.matching

        global Aspect
        global OpinionWord

        Aspect = type("Aspect", (Aspect,), {"negations": negations,
                                            "negation_limit": negation_limit})
        OpinionWord = type("OpinionWord", (OpinionWord,), {"negations": negations,
                                                           "negation_limit": negation_limit})

        aspects, opinion_words = double_propagation(sentences, seeds, Aspect, OpinionWord, matching, key)
        clause_pruning(sentences, aspects, opinion_words, conjunctions)
        product_pruning(sentences, aspects, opinion_words, product_phrases, dealer_phrases)
        phrase_aspects = get_phrase_aspects(aspects, Q, K, Aspect, matching, key)
        aspects += phrase_aspects

        # reset sentences
        for sentence in sentences:
            del sentence.aspects
            del sentence.opinion_words

        return aspects, opinion_words

# ------- MATCHING FUNCTIONS --------------------------------------------------


class ExactMatching(object):

    def __init__(self, key=lambda x: x.string.lower()):
        """
        Strictly compares the given token to term objects. For comparison,
        applies the function given in "key" parameter to the
        Token/tuple of Tokens.

        Input:
            key   : function to apply to the token,
            default=lambda x: x.string.lower()
        """
        self.key = key

    def __call__(self, token_tuple, terms):
        """
        Input:
            token_tuple : Token or tuple of Token objects
            terms       : term or iterable terms of terms to match

        Output:
            Returns None if no match is found.
            Returns the first term that matches the given token.
        """
        if not hasattr(terms, '__iter__'):
            terms = [terms]
        if not isinstance(token_tuple, tuple):
            token_tuple = (token_tuple,)
        try:
            token_tuple = tuple(self.key(token) for token in token_tuple)
        except Exception as e:
            token_tuple = tuple(str(token) for token in token_tuple)

        for term in terms:
            if any([token_tuple == term_i for term_i in term]):
                # print "compared", token, best_term
                return term
        else:
            # print "compared", token, None
            return None


class FuzzyMatching(object):

    def __init__(self, match=90, key=lambda x: x.string.lower()):
        """
        Fuzzy matching between the given token and term objects. For comparison
        applies the function given in the "key" parameter to the Token/tuple
        of Tokens. Parameter match defines the minimum similarity ratio for
        a match when comparing.

        Input:
            match : minimum similarity for fuzzy matching (%)
            key   : function to apply to the token,
                    default=lambda x: x.string.lower()
        """
        self.match = match
        self.key = key

    def __call__(self, token_tuple, terms):
        """
        Input:
            token_tuple : Token or tuple of Token objects
            terms       : term or iterable of terms to match

        Output:
            Returns None if no match is found.
            Returns the first matched in case many of them show the same
            similarity ratio.
        """
        if not hasattr(terms, '__iter__'):
            terms = [terms]
        if not isinstance(token_tuple, tuple):
            token_tuple = (token_tuple,)
        try:
            token_tuple = tuple(self.key(token) for token in token_tuple)
        except Exception:  # as e
            token_tuple = tuple(str(token) for token in token_tuple)

        best_term = None
        best_ratio = 0

        for term in terms:
            ratio = max([Levenshtein.ratio(unicode(" ".join(token_tuple)),
                                           unicode(" ".join(term_i)))*100
                         for term_i in term])
            if ratio >= self.match and ratio > best_ratio:
                best_term = term
                best_ratio = ratio

        return best_term


# ------- UTIL FUNCTIONS ------------------------------------------------------

def equals(rel_a, rel_b):
    """
    Here equivalent specifically means mod is the same as pnmod,
    and s or subj is the same as obj)
    """
    if rel_a == rel_b:
        return True
    elif (rel_a == 'mod' or rel_b == 'mod') \
    and (rel_a == 'pnmod' or rel_b == 'pnmod'):
        return True
    elif (rel_a == 's' or rel_b == 's') \
    and (rel_a == 'subj' or rel_b == 'subj'):
        return True
    elif (rel_a == 'subj' or rel_b == 'subj') \
    and (rel_a == 'obj' or rel_b == 'obj'):
        return True
    elif (rel_a == 'subj' or rel_b == 'subj') \
    and (rel_a == 's' or rel_b == 's'):
        return True
    else:
        return False


def stanford_to_minipar(relation):
    if relation == "amod":
        return "mod"
    elif "prep" in relation:
        return "pnmod"
    # to include collapsed dependencies
    elif "nsubj" in relation:
        return "subj"
    # to include collapsed dependencies
    elif "csubj" in relation:
        return "subj"
    elif relation == "xsubj":
        return "obj"
    elif relation == "dobj":
        return "obj2"
    elif relation == "iobj":
        return "desc"
    # to include collapsed dependencies
    elif "conj" in relation:
        return "conj"
    else:
        return relation

# ------- EXTRACTION RULES ----------------------------------------------------


def apply_rule_1_1(sentence, opinion_words, matching):
    """
    Extracts aspects based on known opinion words and direct dependency
    using the rule: (we don't force the pos-tag of the matched
    token to be in JJ)

    T → O-Dep → O
        s.t
            O ∈ { O }
            O-Dep ∈ { MR }
            POS(T) ∈ { NN }
        then  t = T

    Returns a list of extracted tuples:
         (existing Term, matched Token, extracted Token)
    """
    # if not sentence.is_parsed:
    #    raise TypeError("Sentence is not parsed.")
    aspects = []
    for relation in sentence.relations:
        head = relation.head
        label = stanford_to_minipar(relation.label)
        dependent = relation.dependent
        match = matching(dependent, opinion_words)
        if head.pos in NN \
        and label in MR \
        and match:
            aspects.append((match, dependent, head))
    return aspects


def apply_rule_1_2(sentence, opinion_words, matching):
    """
    Extracts aspects based on known opinion words and indirect dependency using the rule:
    (we don't force the pos-tag of the matched token to be in JJ)

    O ← O-Dep ← H → T-Dep → T
    s.t.
        H any word
        O ∈ { O }
        O/T-Dep ∈ { MR }
        POS(T) ∈ { NN }
    then t = T

    Returns a list of extracted tuples:
         (existing Term, matched Token, extracted Token)

    """
    # if not sentence.is_parsed:
    #    raise TypeError("Sentence is not parsed.")
    aspects = []
    for token in sentence:
        match = matching(token, opinion_words)
        if match \
        and token.head:
            pivot, label_a = token.head
            if stanford_to_minipar(label_a) in MR:
                for dependent, label_b in pivot.dependents:
                    if stanford_to_minipar(label_b) in MR \
                    and dependent is not token  \
                    and dependent.pos in NN:
                        # print token
                        # print 'v ' + label_a
                        # print pivot
                        # print '^ ' + label_b
                        # print dependent
                        # print ""
                        aspects.append((match, token, dependent))
    return aspects


def apply_rule_2_1(sentence, aspects, matching):
    """
    Extracts opinion words based on known aspects and direct dependency.
    (here we force the pos-tag of the matched token to be in NN)

    T → O-Dep → O
    s.t.
        T ∈ { T }
        O-Dep ∈ { MR }
        POS(O) ∈ { JJ }
    Output: o = O

    Returns a list of extracted tuples:
         (existing Term, matched Token, extracted Token)
    """
    # if not sentence.is_parsed:
    #    raise TypeError("Sentence is not parsed.")
    opinion_words = []
    for relation in sentence.relations:
        head = relation.head
        label = stanford_to_minipar(relation.label)
        dependent = relation.dependent
        if head.pos in NN:
            match = matching(head, aspects)
            if match \
            and dependent.pos in JJ \
            and label in MR:
                # print head, label, dependent
                opinion_words.append((match, head, dependent))
    return opinion_words


def apply_rule_2_2(sentence, aspects, matching):
    """
    Extracts opinion words based on known aspects and indirect dependency.
    (here we force the pos-tag of the matched token to be in NN)

    O ← O-Dep ← H → T-Dep → T
    s.t. 
        T ∈ { T } 
        O/T-Dep ∈ { MR }
        POS(O) ∈ { JJ }
    Output: o = O

    Returns a list of extracted tuples:
         (existing Term, matched Token, extracted Token)
    """
    # if not sentence.is_parsed:
    #    raise TypeError("Sentence is not parsed.")
    opinion_words = []
    for token in sentence:
        if token.pos in NN \
        and token.head:
            match = matching(token, aspects)
            if match:
                pivot, label_a = token.head
                if stanford_to_minipar(label_a) in MR:
                    for dependent, label_b in pivot.dependents:
                        if stanford_to_minipar(label_b) in MR \
                        and dependent is not token  \
                        and dependent.pos in JJ:
                            # print token
                            # print 'v ' + label_a
                            # print pivot
                            # print '^ ' + label_b
                            # print dependent
                            # print ""
                            opinion_words.append((match, token, dependent))
    return opinion_words


def apply_rule_3_1(sentence, aspects, matching):
    """
    Extracts aspects using known aspects and the conjunction relationship.
    (here we force the pos-tag of the matched token to be in NN)

    T_i(j) → T_i(j)-Dep → T_j(i)
    s.t.
        T_j(i) ∈ { T }
        T_i(j)-Dep ∈ { CONJ }
        POS(T_i(j)) ∈ { NN }
    Output: t = T_i(j)

    Returns a list of extracted tuples:
         (existing Term, matched Token, extracted Token)
    """
    # if not sentence.is_parsed:
    #    raise TypeError("Sentence is not parsed.")
    new_aspects = []
    for relation in sentence.relations:
        head = relation.head
        if head.pos in NN:
            label = stanford_to_minipar(relation.label)
            dependent = relation.dependent
            if label in CONJ \
            and dependent.pos in NN:
                match = matching(head, aspects)
                if match:
                    # print head, label, dependent
                    new_aspects.append((match, head, dependent))
    return new_aspects


def apply_rule_3_2(sentence, aspects, matching):
    """
    Extracts aspects using known aspects and indirect same class dependency.
    (here we force the pos-tag of the matched token to be in NN)

    T_i ← T_i-Dep ← H → T_j-Dep → T_j 
    s.t
        H any word.string
        T_i ∈ { T }
        T_i -Dep == T_j-Dep
        POS(T_j) ∈ { NN }
    Then, t = T_j

    Returns a list of extracted tuples:
         (existing Term, matched Token, extracted Token)
    """
    # if not sentence.is_parsed:
    #    raise TypeError("Sentence is not parsed.")
    new_aspects = []
    for token in sentence:
        if token.pos in NN \
        and token.head:
            match = matching(token, aspects)
            if match:
                pivot, label_a = token.head
                label_a = stanford_to_minipar(label_a)
                for dependent, label_b in pivot.dependents:
                    label_b = stanford_to_minipar(label_b)
                    if equals(label_a, label_b) \
                    and dependent is not token  \
                    and dependent.pos in NN:
                        # print token
                        # print 'v ' + label_a
                        # print pivot
                        # print '^ ' + label_b
                        # print dependent
                        # print ""
                        new_aspects.append((match, token, dependent))
    return new_aspects


def apply_rule_4_1(sentence, opinion_words, matching):
    """
    Extract opinion words based on known opinion words and the conjunction
    relationship.
    Ex: “Bill is big and honest” conj(big, honest)
    (we don't force the pos-tag of the matched opinion word to be in JJ)

    O_i(j) → O_i(j)-Dep → O_j(i)
    s.t.
        O_j(i) ∈ { O } ,
        O_i(j)-Dep ∈ { CONJ }
        POS(O_i(j)) ∈ { JJ }
    Output: o = O_i(j)

    Returns a list of extracted tuples:
         (existing Token, extracted Token)
    """
    # if not sentence.is_parsed:
    #    raise TypeError("Sentence is not parsed.")
    new_opinion_words = []
    for relation in sentence.relations:
        head = relation.head
        label = stanford_to_minipar(relation.label)
        dependent = relation.dependent
        match = matching(head, opinion_words)
        if label in CONJ \
        and dependent.pos in JJ \
        and match:
            # print head, label, dependent
            new_opinion_words.append((match, head, dependent))
    return new_opinion_words


def apply_rule_4_2(sentence, opinion_words, matching):
    """
    Extract opinion words based on known opinion words and indirect dependency.
    (we don't force the pos-tag of the matched token to be in JJ)

    O_i ← O_i-Dep ← H → O_j-Dep → O_j
    s.t.
        H any word
        O_i ∈ { O }
        O_i-Dep == O_j-Dep
        POS(O_j) ∈ { JJ }
    Output: o = O_j
    """
    # if not sentence.is_parsed:
    #    raise TypeError("Sentence is not parsed.")
    new_opinion_words = []
    for token in sentence:
        match = matching(token, opinion_words)
        if match \
        and token.head:
            pivot, label_a = token.head
            label_a = stanford_to_minipar(label_a)
            for dependent, label_b in pivot.dependents:
                label_b = stanford_to_minipar(label_b)
                if equals(label_a, label_b) \
                and dependent is not token \
                and dependent.pos in JJ:
                    # print token
                    # print 'v ' + label_a
                    # print pivot
                    # print '^ ' + label_b
                    # print dependent
                    # print ""
                    new_opinion_words.append((match, token, dependent))
    return new_opinion_words


def double_propagation(sentences, seeds, Aspect, OpinionWord,
                       matching=ExactMatching(), key=lambda x: x.stem):
    """
    Input:
        sentences   : list of :Sentences
        seeds       : dict of seeds
        Aspect      : class Aspect
        OpinionWord : class OpinionWord
        matching    : matching function
    """

    aspects = []
    opinion_words = []

    for seed in seeds.values():
        match = matching(seed, opinion_words)
        if match:
            match.add_seed(key, seed)
        else:
            opinion_word = OpinionWord(key, seed=seed)
            opinion_words.append(opinion_word)

    # prepare sentences
    for sentence in sentences:
        sentence.aspects = []
        sentence.opinion_words = []

    while True:
        aspects_i = []
        opinion_words_i = []

        # first loop, using all aspects 
        for sentence in sentences:
            # extract aspects using all known opinion words and DD
            for existing, matched, extracted in apply_rule_1_1(sentence, opinion_words, matching):
                do_aspect(aspects, existing, matched, extracted,
                          aspects_i, Aspect, matching, key)

            # extract aspects using all known opinion words and ID
            for existing, matched, extracted in apply_rule_1_2(sentence, opinion_words, matching):
                do_aspect(aspects, existing, matched, extracted,
                          aspects_i, Aspect, matching, key)

            # extract opinion words based on all known opinion words and conjunction
            for existing, matched, extracted in apply_rule_4_1(sentence, opinion_words, matching):
                do_opinion_word(opinion_words, existing, matched, extracted,
                                opinion_words_i, OpinionWord, matching, key)

            # extract opinion words based on all known opinion words and ID
            for existing, matched, extracted in apply_rule_4_2(sentence, opinion_words, matching):
                do_opinion_word(opinion_words, existing, matched, extracted,
                                opinion_words_i, OpinionWord, matching, key)

        aspects += aspects_i
        opinion_words += opinion_words_i

        # print "Iter 1"
        # print(len(aspects_i))
        # print(len(opinion_words_i))
        # print "---"

        aspects_star = []
        opinion_words_star = []

        # second loop, using recently extracted aspects
        for sentence in sentences:
            # extract aspects using recently aspects and conjunction
            for existing, matched, extracted in apply_rule_3_1(sentence, aspects_i, matching):
                do_aspect(aspects, existing, matched, extracted,
                          aspects_star, Aspect, matching, key)

            # extracts aspects using recently extracted aspects and class DD
            for existing, matched, extracted in apply_rule_3_2(sentence, aspects_i, matching):
                do_aspect(aspects, existing, matched, extracted,
                          aspects_star, Aspect, matching, key)

            # extracts opinion words based on recently extracted aspects and DD
            for existing, matched, extracted in apply_rule_2_1(sentence, aspects_i, matching):
                do_opinion_word(opinion_words, existing, matched, extracted,
                                opinion_words_star, OpinionWord, matching, key)

            # extract opinion words based on recently extracted aspects and ID
            for existing, matched, extracted in apply_rule_2_2(sentence, aspects_i, matching):
                do_opinion_word(opinion_words, existing, matched, extracted,
                                opinion_words_star, OpinionWord, matching, key)

        aspects += aspects_star
        opinion_words += opinion_words_star

        aspects_i += aspects_star
        opinion_words_i += opinion_words_star

        # print "Iter 2"
        # print(len(aspects_star))
        # print(len(opinion_words_star))
        # print "---"

        if len(aspects_i) == 0 and len(opinion_words_i) == 0:
            break

    return aspects, opinion_words


def do_aspect(aspects, existing, matched, extracted,
              new_aspects, Aspect, matching, key):
    """
    aspects     : dict of Aspect objects
    existing    : existing Term (OpinionWord or Aspect) used in rule
    matched     : Token matching the Term used in rule
    extracted   : Token extracted by rule
    matching    : matching function
    new_aspects : dict to add new Aspects
    """
    # append the matched Token to the existing Term
    appended = existing.append(key, existing, (matched,))
    extracted = (extracted,)
    # check if we have a "new" Aspect that matches the extracted Token
    new_aspect = matching(extracted, new_aspects)
    if new_aspect:
        # if new_match, add Token to matched "new" Aspect]
        new_aspect.append(key, appended, extracted)
    else:
        # check if we have an existing Aspect that matches the extracted Token
        aspect = matching(extracted, aspects)
        # if matches, add Token to matched Aspect
        if aspect:
            aspect.append(key, appended, extracted)
        else:
            # if new add it to "new" Aspects
            new_aspect = Aspect(key, appended, extracted)
            new_aspects.append(new_aspect)


def do_opinion_word(opinion_words, existing, matched, extracted,
                    new_opinion_words, OpinionWord, matching, key):
    """
    Function that processes the extracted opinion word.
    opinion_words     : dict of OpinionWord objects
    existing          : existing Term (OpinionWord or Aspect) used in rule
    matched           : Token matching the Term used in rule
    extracted         : Token extracted by rule
    matching          : matching function
    new_opinion_words : dict to add new OpinionWords here
    """
    # append the matched Token to the existing Term
    appended = existing.append(key, existing, (matched,))
    extracted = (extracted,)

    # check if we have a "new" OpinionWord that matches the extracted Token
    new_opinion_word = matching(extracted, new_opinion_words)
    if new_opinion_word:
        # if new_match, add Token to matched "new" OpinionWord
        new_opinion_word.append(key, appended, extracted)
    else:
        # check if we have an existing OpinionWord that matches
        # the extracted Token
        opinion_word = matching(extracted, opinion_words)
        # if matches, add Token to matched OpinionWord
        if opinion_word:
            opinion_word.append(key, appended, extracted)
        else:
            # if new add it to "new" OpinionWords
            new_opinion_word = OpinionWord(key, appended, extracted)
            new_opinion_words.append(new_opinion_word)


# ------- ORIENTATION RULES ---------------------------------------------------

# In theory, some functions below can be added as methods to Sentence or
# Document objects in module "reps.py". I decided not to do so to enhance
# clarity and for completeness of the implementation of DoublePropagation

def has_near_negation(token, negations, limit):
    sentence = token.sentence
    pos_1 = token.index-limit if token.index-limit >= 0 else 0
    pos_2 = token.index+limit if token.index+limit <= len(sentence) else len(sentence)
    return negations_between(negations, sentence=sentence,
                             pos_1=pos_1, pos_2=pos_2)


def negations_between(negations, **kwargs):
    """
    Counts the negations between the given tokens/positions
    Only kwargs.
    """
    token_1 = kwargs.get('token_1', None)
    token_2 = kwargs.get('token_1', None)
    sentence = kwargs.get('sentence', None)
    pos_1 = kwargs.get('pos_1', 0)
    pos_2 = kwargs.get('pos_2', None)

    if token_1 and token_2:
        if token_1.sentence is not token_2.sentence:
            raise Exception('Tokens need to belong to the same sentence.')
        sentence = token_1.sentence
        pos_1 = min(token_1.index, token_2.index)
        pos_2 = max(token_1.index, token_2.index)
        return len([token for token in sentence[pos_1:pos_2]
                    if token.string.lower() in negations])
    elif sentence:
        if pos_2 is None:
            pos_2 = len(sentence)
        return len([token for token in sentence[pos_1:pos_2]
                    if token.string.lower() in negations])
    else:
        raise Exception('You have to provide either 2 tokens or a sentence')


def review_orientation(review):
    orientation = 0.0
    for sentence in review.sentences:
        for opinion_word in sentence.opinion_words:
            # print opinion_word
            orientation += opinion_word.orientation
    return 1.0 if orientation > 0 else -1.0

# ------- PRUNING FUNCTIONS ---------------------------------------------------


def has_conjunction(sentence, start, end, conjunctions):
    for word in conjunctions:
        if word in [token.string.lower() for token in sentence[start:end]]:
            return True
    else:
        return False


def max_support(aspects):
    return max(aspects, key=lambda a: a.term.support)


def clean_empty_terms(terms, min=0):
    for index, term in enumerate(terms):
        if term.support <= min:
            terms.pop(index)


def clear_extracted_by_pruned(terms, deletes):
    """Recursively delete objects that were extracted by pruned aspects"""
    new_deletes = []
    while True:
        for term in terms:
            for sentence in term.sentences:
                existing, extracted = sentence.history[0]
                if existing in deletes:
                    delete = sentence.delete()
                    new_deletes.append(delete)
        if new_deletes:
            deletes = new_deletes
        break


def clause_pruning(sentences, aspects, opinion_words, conjunctions):
    deletes = []
    for sentence in sentences:
        sentence_aspects = sentence.aspects
        if len(sentence_aspects) > 1:
            clauses = sentence.clauses
            if len(clauses) == 0:
                max_aspect = max_support(sentence_aspects)
                for aspect in sentence_aspects:
                    if aspect != max_aspect:
                        delete = aspect.delete()
                        deletes.append(delete)
                sentence.aspects = [max_aspect]

            else:
                positions = [0]
                for clause in clauses:
                    positions += [clause.start, clause.stop]
                positions += [len(sentence)]
                # reset sentence.aspects
                sentence.aspects = []
                for i in range(len(positions)-1):
                    start, stop = positions[i:i+2]
                    segment_aspects = [aspect for aspect in sentence_aspects
                                      if start <= aspect.positions[0] < stop] 
                    if len(segment_aspects) > 1 and not has_conjunction(sentence, start, stop, conjunctions):
                        max_aspect = max_support(segment_aspects)
                        for aspect in segment_aspects:
                            if aspect != max_aspect:
                                delete = aspect.delete()
                                deletes.append(delete)
                        sentence.aspects.append(max_aspect)

    clean_empty_terms(aspects)
    clear_extracted_by_pruned(aspects, deletes)
    clear_extracted_by_pruned(opinion_words, deletes)
    clean_empty_terms(aspects)
    clean_empty_terms(opinion_words)


def product_pruning(sentences, aspects, opinion_words,
                    product_phrases, dealer_phrases):
    deletes = []
    for sentence in sentences:
        for phrase in product_phrases:
            hit = sentence.contains(phrase, key=lambda x: x.stem)
            if hit:
                # get the nearest noun folowing the prhase
                start, stop = hit
                nouns = [noun for noun in sentence.nouns if noun.index > stop]
                if nouns:
                    if len(nouns) > 1:
                        noun = min(nouns, key=lambda n: abs(n.index-stop))
                    else:
                        noun = nouns[0]
                else:
                    noun = None
                # do pruning
                if noun:
                    for aspect in sentence.aspects:
                        if aspect.tokens[0] is noun:
                            delete = aspect.delete()
                            deletes.append(delete)

        for phrase in dealer_phrases:
            hit = sentence.contains(phrase, key=lambda x: x.stem)
            if hit:
                # get the nearest noun before the phrase
                start, stop = hit
                nouns = [noun for noun in sentence.nouns if noun.index < start]
                if nouns:
                    if len(nouns) > 1:
                        noun = min(nouns, key=lambda n: abs(n.index-start))
                    else:
                        noun = nouns[0]
                else:
                    noun = None
                # do pruning
                if noun:
                    for aspect in sentence.aspects:
                        if aspect.tokens[0] is noun:
                            delete = aspect.delete()
                            deletes.append(delete)

    clean_empty_terms(aspects)
    clear_extracted_by_pruned(aspects, deletes)
    clear_extracted_by_pruned(opinion_words, deletes)
    clean_empty_terms(aspects)
    clean_empty_terms(opinion_words)


def get_phrase_aspects(aspects, Q, K, Aspect, matching, key):

    transactions = {}
    existings = defaultdict(list)

    for aspect in aspects:
        for ais in aspect.sentences:
            existing, extracted = ais.history[0]
            sentence_id = ais.sentence.id
            # use first position
            for token in ais.tokens:
                term = token[0]
                post_nouns = sorted([noun for noun in ais.sentence.nouns
                                     if noun.index > term.index],
                                    key=lambda x: abs(x.index-term.index))
                prev_nouns = sorted([noun for noun in ais.sentence.nouns
                                     if noun.index < term.index],
                                    key=lambda x: abs(x.index-term.index))
                prev_adjs = sorted([adj for adj in ais.sentence.adjectives
                                    if adj.index < term.index],
                                   key=lambda x: abs(x.index-term.index))
                phrases = set()
                if prev_nouns:
                    for i in range(1, Q+1):
                        phrase = sorted([term]+prev_nouns[:i], key=lambda x: x.index)
                        pphrase = tuple(key(token) for token in phrase)
                        phrases.add(pphrase)
                        existings[(pphrase, sentence_id)].append((existing, tuple(phrase)))

                if post_nouns:
                    for i in range(1, Q+1):
                        phrase = sorted([term]+post_nouns[:i], key=lambda x: x.index)
                        pphrase = tuple(key(token) for token in phrase)
                        phrases.add(pphrase)
                        existings[(pphrase, sentence_id)].append((existing, tuple(phrase)))
                if prev_adjs:
                    for i in range(1, K+1):
                        phrase = sorted([term]+prev_adjs[:i], key=lambda x: x.index)
                        pphrase = tuple(key(token) for token in phrase)
                        phrases.add(pphrase)
                        existings[(pphrase, sentence_id)].append((existing, tuple(phrase)))
                if phrases:
                    ps = transactions.get(ais.sentence.id, None)
                    if ps:
                        ps |= phrases
                    else:
                        transactions[ais.sentence.id] = phrases

    new_aspects = []
    frequents = fim(transactions.values(), supp=2, zmax=1)
    for frequent, support in frequents:
        frequent = frequent[0]
        for id_sentence, transaction in transactions.items():
            if frequent in transaction:
                existing, extracted = existings[(frequent, id_sentence)][0]
                aspect = matching(frequent, new_aspects)
                if aspect:
                    aspect.append(key, existing, extracted)
                else:
                    aspect = Aspect(key, existing, extracted)
                    new_aspects.append(aspect)
    return new_aspects


# ------- BASE CLASSES --------------------------------------------------------

class Term(object):

    def __init__(self, existing, extracted):
        raise NotImplementedError()

    def __eq__(self, other):
        if type(self) == type(other) and self.id == other.id:
            return True
        return False

    @property
    def tiss(self):
        # dict of term in sentence Objects
        all_items = [(key, value) for review in self.reviews
                     for (key, value) in review.tiss.items()]
        return {key: value for (key, value) in all_items}

    @property
    def reviews(self):
        return self.tirs.values()

    @property
    def sentences(self):
        # dict of term in sentence Objects
        return [sentence for review in self.reviews
                for sentence in review.sentences]

    @property
    def sentence_ids(self):
        return self.tiss.values()

    @property
    def review_ids(self):
        return self.tirs.values()

    @property
    def support(self):
        return len(self.tiss)

    def in_sentence(self, sentence_id):
        return self.tiss[sentence_id]

    def in_review(self, review_id):
        return self.tirs[review_id]


class TermInReview(object):

    def __init__(self):
        raise NotImplementedError()

    def __repr__(self, *args, **kwargs):
        return self.__str__()

    @property
    def sentences(self):
        return self.tiss.values()

    @property
    def id(self):
        return self.term.id, self.review.id

    @property
    def support(self):
        return len(self.tiss)

    def delete(self):
        # print "deleting " + str(self)
        del self.term.tirs[self.review.id]

    def __eq__(self, other):
        if type(self) == type(other) and self.id == other.id:
            return True
        return False


class TermInSentence(object):

    def __init__(self, term_in_review, existing, extracted):
        self.sentence = extracted.sentence
        self.tir = term_in_review
        self.tokens = [extracted]

        # history keeps track of extraction process
        self.history = []
        self.history.append((existing, extracted))

    def __repr__(self, *args, **kwargs):
        return self.__str__()

    def __eq__(self, other):
        if type(self) == type(other) and self.id == other.id:
            return True
        return False

    @property
    def positions(self):
        return [token.index for token in self.tokens]

    @property
    def review(self):
        return self.tir

    @property
    def term(self):
        return self.tir.term

    @property
    def id(self):
        return self.term.id, self.sentence.id

    def append(self, existing, extracted):
        # print "appending to" + str(self) 
        if extracted not in self.tokens:
            self.tokens.append(extracted)
        self.history.append((existing, extracted))
        return self

    def delete(self):
        # print "deleting " + str(self)
        deleted = self.review.tiss[self.sentence.id]
        del self.review.tiss[self.sentence.id]
        if self.review.support == 0:
            self.review.delete()
        return deleted

# ------- ASPECT CLASS --------------------------------------------------------


class Aspect(Term):

    negations = NEGATIONS
    negation_limit = NEGATION_LIMIT

    def __init__(self, key, existing, extracted):
        # dict of term in review Objects
        self.id = _uid()
        self.set = set([tuple(key(token) for token in extracted)])
        review_id = extracted[0].sentence.review.id
        self.is_compound = False

        if len(extracted) > 1:
            self.is_compound = True

        self.tirs = OrderedDict()
        self.tirs[review_id] = Aspect.AspectInReview(self, existing, extracted)

    def append(self, key, existing, extracted):
        self.set.add(tuple(key(token) for token in extracted))
        review_id = extracted[0].sentence.review.id

        tir = self.tirs.get(review_id, None)
        if tir:
            tis = tir.append(existing, extracted)
            return tis
        else:
            tir = Aspect.AspectInReview(self, existing, extracted)
            self.tirs[review_id] = tir
            return tir.tiss.values()[0]

    def __str__(self, *args, **kwargs):
        return "Aspect<{0}>".format(self.id)

    def __repr__(self, *args, **kwargs):
        return self.__str__()

    def __iter__(self):
        return iter(self.set)

    # ============================================================================================================

    class AspectInReview(TermInReview):

        def __init__(self, term, existing, extracted):
            self.term = term
            self.is_compound = term.is_compound
            self.review = extracted[0].sentence.review
            sentence_id = extracted[0].sentence.id
            # dict of term in sentence Objects
            self.tiss = OrderedDict()
            self.tiss[sentence_id] = Aspect.AspectInSentence(self, existing, extracted)

        def __str__(self, *args, **kwargs):
            return "Aspect<{0} in Review {1}>".format(*self.id)

        def append(self, existing, extracted):
            """
            Returns the AspectInSentence that was created
            when this function is called
            """
            sentence_id = extracted[0].sentence.id
            tis = self.tiss.get(sentence_id, None)
            if tis:
                tis = tis.append(existing, extracted)
                return tis
            else:
                tis = Aspect.AspectInSentence(self, existing, extracted)
                self.tiss[sentence_id] = tis
                return tis

        @property
        def orientation(self):
            """
            Orientation of an Aspect in review is obtained from the orientation
            of all the related AspectInSentence that were extracted either using
            other OpinionWordInSentence or other AspectInSentence.
            """
            orientation = 0.0
            for sentence in self.sentences:
                # get the "origin" of the AspectInSentence
                existing, extracted = sentence.history[0]
                if existing is not self.term:
                    orientation += sentence.orientation
            return 1.0 if orientation > 0 else -1.0

    # =============================================================================================================

    class AspectInSentence(TermInSentence):

        def __init__(self, term_in_review, existing, extracted):
            """
            Represents an Aspect that appears on a specific sentence.
            It may appear on different locations and each position may be extracted
            several times. set([ tuple(key(seed),) ])
            Extraction history is stored in the history attribute.
            """
            self.is_compound = term_in_review.is_compound
            self.sentence = extracted[0].sentence
            self.tir = term_in_review
            self.tokens = [extracted]

            # histoy keeps track of the
            self.history = []
            self.history.append((existing, extracted))

            self.sentence.aspects += [self]
            # print "creating " + str(self)

        def __str__(self, *args, **kwargs):
            return "Aspect<{0} in Sentence {1}>".format(*self.id)

        @property
        def orientation(self):
            """
            We use self.history and orientation
            rules to determine the orientation
            """
            if len(self.history) > 1:
                # do something maybe?
                pass

            # get the first item in history
            existing, extracted = self.history[0]

            # case 0: AspectInSentence extracted by self matching the Aspect
            if isinstance(existing, Aspect):
                # get orientation of Aspect in the current review
                return existing.in_review(self.review.review.id).orientation

            # case 1: aspect was extracted using known OpinionWordInSentence
            if isinstance(existing, OpinionWord.OpinionWordInSentence):
                # get orientation of OpinionWord (parent of OpinionWordInSentence)
                orientation = existing.term.orientation
                # check if there are negations near the OpinionWordInSentence
                # (we use the first position in which it was found)
                if has_near_negation(existing.tokens[0][0], Aspect.negations, Aspect.negation_limit):
                        return -1.0 * orientation
                return orientation

            # case 2: aspect extracted from known AspectInSentence
            if isinstance(existing, Aspect.AspectInSentence):
                # get the orientation of current AspectInReview
                orientation = existing.review.orientation
                if orientation:
                    n = negations_between(Aspect.negations,
                                          token_1=existing.tokens[0][0],
                                          token_2=extracted[0])
                    orientation *= ((-1.0)**n)
                return orientation

            raise Exception("Class matching failed")

# ------- OPINIONWORD CLASS ---------------------------------------------------


class OpinionWord(Term):

    negations = NEGATIONS
    negation_limit = NEGATION_LIMIT

    def __init__(self, key, existing=None, extracted=None, seed=None):
        # print "creating " + str(self)
        self.id = _uid()
        if seed:
            self.set = set([(key(seed),)])
            self.tirs = OrderedDict()
            self.is_seed = True
            self._orientation = seed.orientation
        else:
            # dict of term in review Objects
            self.set = set([tuple(key(token) for token in extracted)])
            review_id = extracted[0].sentence.review.id
            self.tirs = OrderedDict()
            self.tirs[review_id] = OpinionWord.OpinionWordInReview(self, existing, extracted)
            self.is_seed = False

    def add_seed(self, key, seed):
        self.set.add((key(seed),))

    @property
    def orientation(self):
        if self.is_seed:
            return self._orientation
        else:
            # return overall orientation over the corpus
            orientation = 0.0
            for sentence in self.sentences:
                # get the "origin" of the OpinionWordInSentence
                existing, extracted = sentence.history[0]
                if existing is not self:
                    orientation =+ sentence.orientation
            return 1.0 if orientation > 0 else -1.0

    def __str__(self, *args, **kwargs):
        return "OpinionWord<{0}>".format(self.id)

    def __repr__(self, *args, **kwargs):
        return self.__str__()

    def __iter__(self):
        return iter(self.set)

    def append(self, key, existing, extracted):
        self.set.add(tuple(key(token) for token in extracted))
        review_id = extracted[0].sentence.review.id
        tir = self.tirs.get(review_id, None)
        if tir:
            tis = tir.append(existing, extracted)
            return tis
        else:
            tir = OpinionWord.OpinionWordInReview(self, existing, extracted)
            self.tirs[review_id] = tir
            return tir.tiss.values()[0]

    # ==================================================================================================================

    class OpinionWordInReview(TermInReview):

        def __init__(self, term, existing, extracted):
            self.term = term
            self.review = extracted[0].sentence.review
            id_sentence = extracted[0].sentence.id

            # dict of term in sentence Objects
            self.tiss = OrderedDict()
            self.tiss[id_sentence] = OpinionWord.OpinionWordInSentence(self, existing, extracted)

        def __str__(self, *args, **kwargs):
            return "OpinionWord<{0} in Review {1}>".format(*self.id)

        def append(self, existing, extracted):
            """
            Returns the TermInSentence that was created
            when this function is called
            """
            id_sentence = extracted[0].sentence.id
            tis = self.tiss.get(id_sentence, None)
            if tis:
                tis = tis.append(existing, extracted)
                return tis
            else:
                tis = OpinionWord.OpinionWordInSentence(self, existing, extracted)
                self.tiss[id_sentence] = tis
                return tis

    # =======================================================================================================================

    class OpinionWordInSentence(TermInSentence):

        def __init__(self, term_in_review, existing, extracted):
            self.sentence = extracted[0].sentence
            self.tir = term_in_review
            self.tokens = [extracted]

            # histoy keeps track of the 
            self.history = []
            self.history.append((existing, extracted))

            self.sentence.opinion_words.append(self)

        def __str__(self, *args, **kwargs):
            return "OpinionWord<{0} in Sentence {1}>".format(*self.id)

        @property
        def orientation(self):
            """
            We use self.history and orientation
            rules to determine the orientation
            """
            if self.term.is_seed:
                return self.term.orientation

            if len(self.history) > 1:
                # do something maybe?
                pass

            # get the first item in history
            existing, extracted = self.history[0]

            # case 0: OpinionWordInSentence extracted matching from OpinionWord
            if isinstance(existing, OpinionWord):
                # we just return the orientation of the OpinionWord
                return existing.orientation

            # case 1: OpinionWord was extracted using other OpinionWordInSentence
            if isinstance(existing, OpinionWord.OpinionWordInSentence):
                orientation = existing.term.orientation
                if orientation:
                    # count the negations between the OpinionWordInSentences
                    # for the existing one, we use its first-found position
                    n = negations_between(OpinionWord.negations,
                                          token_1=existing.tokens[0][0],
                                          token_2=extracted[0])
                    orientation *= ((-1.0)**n)
                return orientation

            # case 2: OpinionWord was extracted using a AspectInSentence
            if isinstance(existing, Aspect.AspectInSentence):
                # check if AspectInSentence has orientation
                orientation = existing.orientation
                if orientation:
                    # if orientation is not 0
                    # check if there are negations near this OpinionWordInSentence
                    # we get the position where it was first-found
                    if has_near_negation(extracted[0], OpinionWord.negations, OpinionWord.negation_limit):
                        orientation = -1.0 * orientation
                    return orientation
                else:
                    # if AspectInSentence has no orientation
                    # get the orientation of the review, based on OpinionWords
                    review = self.review.review
                    return review_orientation(review)

            raise Exception("Class matching failed")
