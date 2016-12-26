#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from collections import defaultdict

from ..rep import Token

# --- CONSTRAINT FUNCTIONS-----------------------------------------------------


def monolingual(e, f):
    global null
    if f is null:
        return 1.0
    else:
        if e.index == f.index:
            return 0.0
        else:
            return 1.0


def opinion(e, f):
    global null
    if f is null:
        return 1.0
    else:
        if e.index == f.index:
            return 0.0
        elif e.pos.startswith("NN") \
        and f.pos.startswith("JJ"):
            return 1.0
        elif f.pos.startswith("NN") \
        and e.pos.startswith("JJ"):
            return 1.0
        else:
            return 0.0

# --- HELPERS -----------------------------------------------------------------

# represents the "Null" word in the alignment models 
null = Token(None, 'None', index=None)


def normal(token):
    """describes how to normalize a token"""
    return token.string.lower()


def nulled(sentence):
    """Returns the tokens of a sentence plus the "null" Token"""
    global null
    return [null] + sentence.tokens


class Counts(object):
    """
    Object that collects the counts later used to obtain
    probabilities for each model
    """

    def __init__(self):
        self.PROB_SMOOTH = 0.1

        # c(f_i)
        # Koehn total_t
        self.word = defaultdict(int)

        # c(e_j|f_i)  
        # Koehn count_t
        self.words = defaultdict(int)

        # For model 2

        # c(j, l, m) 
        # Koehn total_a
        self.align = defaultdict(int)

        # c(i|j, l, m) 
        # Koehn count_a
        self.aligns = defaultdict(int)

        # For model 3
        # If there is not an initial value, it throws an exception of 
        # the number divided by zero. And the value of computing 
        # probability will be always zero.

        # c(i, l , m)
        # Koehn count_d
        self.distort = defaultdict(lambda: self.PROB_SMOOTH)

        # c(j|i, l, m)
        # Koehn total_d 
        self.distorts = defaultdict(lambda: self.PROB_SMOOTH)

        # Koehn total_f
        self.fertility = defaultdict(lambda: self.PROB_SMOOTH)

        # Koehn count_f
        self.fertilities = defaultdict(lambda: self.PROB_SMOOTH)

        # Koehn count_p1
        self.p1 = 0.5

        # Koehn count_p0
        self.p0 = 0.5


def EM(aligner):
    """Function for computing the expected counts
    using the Expectation Maximization algorithm
    """
    counts = aligner.counts
    # bitext is (source_sentences, target_sentences)
    # source = FROM = FOREING (f)
    # target = TO   = ENGLISH (e)
    for s, (foreign, english) in enumerate(aligner.bitext):
        foreign = nulled(foreign)
        l_e = len(english)
        l_f = len(foreign)
        if isinstance(aligner, IBM1):
            for e in english:
                # s-total e_j
                total = sum([aligner.p(e, f, l_e, l_f) for f in foreign])
                for f in foreign:
                    c = aligner.p(e, f, l_e, l_f) / total
                    counts.word[normal(f)] += c
                    counts.words[normal(e), normal(f)] += c
                    if isinstance(aligner, IBM2):
                        counts.aligns[(f.index, e.index, l_e, l_f)] += c
                        counts.align[(e.index, l_e, l_f)] += c
    aligner.first_run = False

# --- CLASSES -----------------------------------------------------------------


class Aligner(object):
    pass


class IBM1(Aligner):

    def __init__(self, constraint=None):
        "Initialize the alignment model"
        self.first_run = True
        self.constraint = constraint

    def estimate(self, source_sentences, target_sentences, iterations=3):
        """Initializes the counts n(e), and later calls EM algorithm."""
        self._initialize(source_sentences, target_sentences)
        self._estimate(iterations=iterations)

    def _initialize(self, source_sentences, target_sentences):
        self.counts = Counts()
        self.bitext = zip(source_sentences, target_sentences)
        # source = FROM = FOREIGN (f)
        # target = TO   = ENGLISH (e)
        for foreign, english in self.bitext:
            foreign = nulled(foreign)
            # for each target sentence
            for e in english:
                # for each source sentence (adding the NULL word)
                for f in foreign:
                    key = (normal(e), normal(f))
                    if key not in self.counts.words:
                        self.counts.words[key] = 1.0
                        self.counts.word[normal(f)] += 1.0

    def _estimate(self, iterations):
        # EM algorithm. Relies on Counter onject to do the heavy lifting.
        for _ in range(iterations):
            EM(self)

    def align(self, source, target):
        """Computes the best (constrained) alignment."""
        # source = FROM = FOREIGN (f)
        # target = TO   = ENGLISH (e)
        foreign = nulled(source)
        english = target
        l_e = len(english)
        l_f = len(foreign)
        alignment = []
        for e in english:
            max_f, max_p = max([ (f, self.p(e, f, l_e, l_f)) for f in foreign],
                               key=lambda x: x[1])
            alignment.append(max_f)
        return alignment

    def t(self, e, f):
        """The translation parameter t(e|f), common to all models."""
        return self.counts.words[normal(e), normal(f)] / self.counts.word[normal(f)]

    def p(self, e, f, l_e, l_f):
        """
        The alignment function for Model 1 is simply the translation
        probability t(e|f) in parent class AlignmentModel.
        """
        value = self.t(e, f)
        if self.constraint:
            return value*self.constraint(e, f)
        else:
            return value


class IBM2(IBM1):

    def __init__(self, constraint=None, ibm1=None):
        """Initialize the alignment model"""
        self.first_run = True
        if ibm1:
            self.counts = ibm1.counts
            self.constraint = ibm1.constraint
            self.bitext = ibm1.bitext
        else:
            self.constraint = constraint
            self.counts = None

    def estimate(self, source_sentences, target_sentences, iterations=3):
        if self.counts is None:
            ibm1 = IBM1(constraint=self.constraint)
            ibm1.estimate(source_sentences, target_sentences,
                          iterations=iterations)
            self.counts = ibm1.counts
            self.bitext = ibm1.bitext
        self._estimate(iterations)

    def q(self, f, e, l_e, l_f):
        """
        The alignment probability parameter a(i|j,l,m) for Koehn
        The first time through this parameter is just 1/ (l_f) (null token
        already added).
        """
        if self.first_run:
            return 1.0 / (l_f)
        return self.counts.aligns[f.index, e.index, l_e, l_f] / self.counts.align[e.index, l_e, l_f]

    def p(self, e, f, l_e, l_f):
        """
        The alignment function for model 2, includes the translation
        and alignment probailities.
        """
        value = self.t(e, f) * self.q(f, e, l_e, l_f)
        if self.constraint:
            return value*self.constraint(e, f)
        else:
            return value

# class Model3(Model2):
#
#     def __init__(self, model2):
#         self.number = 3
#         self.counts = model2.counts
#
#     def q(self, i, j, l, m):
#         """
#         The alignment probability parameter a(i|j,l,m) 
#         from Model 2, needed for sampling function
#         """
#         return self.counts.aligns[(i, j, l, m)] / self.counts.align[(j, l, m)]
#
#
#     def d(self, j, i, l, m):
#         """
#         Distortion probability d(j|i,le,lf)
#
#          for all (i,j,le,lf) in domain( countd ):
#             d(j|i,le,lf) = countd(j|i,le,lf) / totald(i,le,lf)
#         """
#         return self.counts.distorts[(j, i, l, m)] / self.counts.distort[(i, l, m)]
#
#
#     def n(self, phi, f_i):
#         """
#         Fertility probability n(φ,f_i)
#
#         for all (φ,f) in domain( countf ):
#             n(φ|f) = countf(φ|f) / totalf(f)
#         """
#         return self.counts.fertilities[(phi, f_i)] / self.counts.fertility[f_i]
#
#
#     def p1(self):
#         return self.counts.p1 / (self.counts.p1 + self.counts.p0)
#
#
#     def p0(self):
#         return self.counts.p0 / (self.counts.p1 + self.counts.p0)
#
#     def sample(self, e, f):
#         """
#         Finds the best Model2 Alignment a* and generates
#         the initial fertilities of each word phi*_i
#
#         Using these two pieces, calls hillclimb(e,f,a*,phi*_i )
#         This function gets some neighboring alignments and finds 
#         out the alignment a'* (and the phis) with highest probability 
#         in those alignment spaces.
#
#         Finally returns the neighborhood of a'* as tuples (a,phis)
#         """
#         l = len(e) 
#         m = len(f) 
#         A = []
#         print "Sampling..."
#         print ("English", e)
#         print ("Foreign", f)
#
#         b_a = []
#         for j, e_j in enumerate(e):
#             b_a.append(argmax([(ii, self.t(e_j, f_ii) *self.q(ii, j, l, m)) for ii, f_ii in enumerate(f)])[0])
#
#         print("Best M2 Alignment: ", b_a)
#
#         for i, f_i in enumerate(f):
#
#             i_pegged = i
#
#             for j, e_j in enumerate(e):
#
#                 j_pegged = j
#
#                 # initialize empty alignment and phi's in 0
#                 a = []
#                 phis = [0] * m
#
#                 # find the best Model 2 alignment
#                 for jj, e_jj in enumerate(e):
#
#                     # except for pegged a[j] = i
#                     if  jj == j_pegged:
#                         a.append(i_pegged)
#                         # increase fertility of f_i
#                         phis[i_pegged] = +1
#
#                     else:
#                         (ii, p) = argmax([(ii, self.t(e_jj, f_ii)*self.q(ii, jj, l, m)) for ii, f_ii in enumerate(f)])
#                         a.append(ii)
#                         # increase fertility of f_i
#                         phis[ii] = +1
#
#                 #print "Pegging a[" + str(j_pegged) + "] = " + str(i_pegged)
#                 #print("a:", a)
#                 #print("phis:", phis)
#
#                 #print("HILL CLIMBING")
#
#                 a_star, phis_star = self.hillclimb(a, phis, j_pegged, e, f)
#
#                 #print("Hillclimbed Star")
#                 #print("a_star:", a_star)
#                 #print("phis_star:", a_star)
#
#                 for (neighbor_a,neighbor_phis) in self.neighboring(a_star, phis_star, j_pegged, e, f):
#                     if neighbor_a not in [item[0] for item in A]:
#                         A.append((neighbor_a, neighbor_phis))
#
#         print "Sampling Done"
#
#         return A
#
#     def hillclimb(self, a, phis, j_pegged, e, f):
#         """
#         The hillclimb algorithm.
#         """
#         print("hillclimbing")
#         while True:
#             a_old = a
#             neighbors = self.neighboring(a, phis, j_pegged, e, f)
#             for (neighbor_a, neighbor_phis) in neighbors:
#                 if self.p(neighbor_a, neighbor_phis, e ,f) > self.p(a, phis, e, f): 
#                     a = neighbor_a
#                     phis = neighbor_phis
#
#             # Until this alignment is the highest one in local
#             if a == a_old:
#                 break
#
#         return (a, phis)
# 
#     def neighboring(self, a, phis, j_pegged, e, f):
#         """
#         Getting the neighboring. Returns alignments
#         in the form of tuples (alignment, fertility)
#         """
#         l = len(e) 
#         m = len(f)
#
#         #print("Creating Change Neighbors")
#
#         ## init the of neighboring alignments: (a,phis)
#         N = [] 
#
#         # do moves
#         for j in range(l): 
#
#             # ommit j = j_pegged
#             if j != j_pegged: 
#
#                 for i in range(m): 
#                     new_a = a[:]
#                     new_a[j] = i
#                     new_phis = phis[:]
#                     if new_phis[a[j]] > 0:
#                         # replace fertilies
#                         new_phis[a[j]] -= 1
#                         new_phis[i] += 1
#
#                     #print("new neighbor a", new_a)
#                     #print("new neighbor phis", new_phis)
#
#                     if new_a not in [item[0] for item in N]:
#                         N.append((new_a, new_phis))
#
#
#         # print("Creating Swap Neighbors")
#         # do swaps
#         for j_1 in range(len(e)): 
#
#             # ommit j_1 = j_pegged
#             if j_1 != j_pegged: 
#
#                 for j_2 in range(len(e)):
#
#                     # ommit j_2 = j_pegged
#                     # if we are not swapping the same js
#                     if j_2 != j_pegged and j_1 != j_2:
#                         new_a = a[:]
#                         new_phis = phis[:]
#                         new_a[j_1] = a[j_2] 
#                         new_a[j_2] = a[j_1]
#
#                         # we dont change fertilites
#                         # since we interchange
#
#                         #print("new neighbor a", new_a)
#                         #print("new neighbor phis", new_phis)
#
#                         if new_a not in [item[0] for item in N]:
#                             N.append((new_a, new_phis))
#
#         return N
#
#     def p(self, a, phis, e, f):
#         """
#         Probability function of model 3.
#         This function returns the probability given an alignment.
#         The Fert variable is math syntax 'Phi' in the fomula, which
#         represents the fertility according to the current alignment,
#         which records how many output words are generated by each
#         input word.
#         """
#         l = len(e)
#         m = len(f)
#
#         total = 1.0
#
#         # Compute the NULL insertation
#         total *= pow(self.p1(), phis[0]) * pow(self.p0(), l - 2 * phis[0])
#         if total == 0:
#             print "Zero 1"
#             return total
#
#         # Compute the combination (l - phis[0]) choose phis[0]
#         for i in range(1, phis[0] + 1):
#             total *= (l - phis[0] - i + 1) / i
#             if total == 0:
#                 print "Zero 2"
#                 return total
#
#         # Compute fertilities term (not include phis[0])
#         for i in range(1, m):
#             total *= factorial(phis[i]) * self.n(phis[i],f[i])
#             if total == 0:
#                 print "Zero 3"
#                 return total
#
#         # Multiply the lexical and distortion probabilities
#         for j, e_j in enumerate(e):
#
#             # a[j] is i
#             f_a_j = f[a[j]]
#
#             total *= self.t(e_j, f_a_j)
#             if total == 0:
#                 print "Zero 4 t"
#                 return total
#
#             total *= self.d(j, a[j], l, m)
#             if total == 0:
#                 print "Zero 4 d"
#                 return total
# 
#         return total
