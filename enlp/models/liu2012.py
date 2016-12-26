#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict
import numpy as np
from numpy import linalg as LA

from enlp.idf import Wikipedia, GoogleBooks
from enlp.stopwords import StanfordStopwords
from enlp.cvalue import generate_multi_word_terms
from enlp.rep import Sentence
from enlp.pipe.aligners import IBM1, IBM2, opinion, null
from .utils import LowerStringKey

LAMB = 0.3

# ------- MAIN CLASS ---------------------------------------------------------------------------------------------------


class GraphTargetExtractor(object):
    
    def __init__(self,
                 limit=100,
                 aligner="IBM2",
                 grouping="simple",
                 lamb=LAMB,
                 idf="wikipedia",
                 stopwords=StanfordStopwords(),
                 condition=2,
                 max_ngram_len=4,
                 key=LowerStringKey()):
        """
        Input:
            - limit          : int, when using "ngram" or "condition" group extraction, limit of 
                               multi-word terms to extract, default=100
            - aligner        : string, "IBM1" or "IBM2", aligner model, default="IBM2" 
            - grouping       : string,  "ngram" or "simple" or "condition"
            - lambda         : int, mixture for random walk, default=LAMB
            - idf            : string, source to calculate idf of a term, either "wikipedia" 
                               or "google", default="wikipedia"
            - stopwords      : iterable, stoplist, default=StanfordStopwords
            - condition      : string, for when using grouping option condition", either 
                               1, 2 or 3, default=2
            - max_ngram_len  : int, when using grouping option "ngram", maximum length of the ngrams 
                               to extract, default=4
            - key            : function to apply to Token when extracting groups, default=lambda x: x.string
        """
        self.params = {"limit": limit,
                       "aligner": aligner,
                       "grouping": grouping,
                       "lambda": lamb,
                       "idf": idf,
                       "stopwords": stopwords,
                       "condition": condition,
                       "max_ngram_len": max_ngram_len,
                       "key": key}
    
    def extract(self, sentences):
        """
        Input:
            - sentences: list of sentences.
        Output:
            - list extracted terms (strings), sorted by confidence.
        """
        lamb = self.params["lambda"]
        
        if self.params["aligner"] == "IBM1":
            aligner = IBM1(constraint=opinion)
        elif self.params["aligner"] == "IBM2":
            aligner = IBM2(constraint=opinion)
        
        if self.params["idf"] == "google":
            idf_calculator = GoogleBooks()
        elif self.params["idf"] == "wikipedia":
            idf_calculator = Wikipedia()
        
        new_sentences = []
        
        if self.params["grouping"] == "simple":
            new_sentences = simple_group_nps(sentences)
        
        elif self.params["grouping"] == "ngram":
            terms = generate_multi_word_terms(sentences, 
                                              method="ngram", 
                                              max_ngram_len=self.params["max_ngram_len"], 
                                              limit=self.params["limit"],
                                              stopwords=self.params["stopwords"],
                                              key=self.params["key"])
            new_sentences = group_sentences(sentences, terms)                                                  
        
        elif self.params["grouping"] == "condition":
            terms = generate_multi_word_terms(sentences, 
                                              method="condition", 
                                              condition=self.params["condition"],
                                              limit=self.params["limit"],
                                              stopwords=self.params["stopwords"],
                                              key=self.params["key"])
            
            new_sentences = group_sentences(sentences, terms)
        
        aligner.estimate(new_sentences, new_sentences)
        
        # tuple order : NN, JJ
        pairs = defaultdict(int) 
        alignment_matrix = defaultdict(set)
        
        for new_sentence in new_sentences:
            aligned_tokens = aligner.align(new_sentence, new_sentence)
    
            for token, aligned_token in zip(new_sentence, aligned_tokens): 
                if null not in [token, aligned_token]:
                    if token.pos.startswith('NN'):
                        if aligned_token.pos.startswith('JJ'):    
                            pairs[token.string.lower(), aligned_token.string.lower()] += 1.0
                            alignment_matrix[token.string.lower(), aligned_token.string.lower()].add(new_sentence.id) 
                        else:
                            # print token, aligned_token
                            pass
                    if token.pos.startswith('JJ'):
                        if aligned_token.pos.startswith('NN'):
                            pairs[aligned_token.string.lower(), token.string.lower()] += 1.0
                            alignment_matrix[aligned_token.string.lower(), token.string.lower()].add(new_sentence.id)
                        else:
                            # print token, aligned_token
                            pass

        nn_counts, jj_counts = single_counts(pairs)
        assoc_probs = association_probs(pairs, nn_counts, jj_counts)
        
        targets = set([pair[0] for pair in assoc_probs.keys()])
        opinion_words = ([pair[1] for pair in assoc_probs.keys()])
        
        targets = list(targets)
        opinion_words = list(opinion_words)
        
        idfs = idf_calculator.idfs(targets)
        importance = {}

        for target, idf in zip(targets, idfs):
            importance[target] = calc_frequency(target, new_sentences)*idf
        
        # define word keys
        key2target = {i: target for i, target in enumerate(targets)}
        target2key = {target: i for i, target in key2target.iteritems()}

        key2opinion_word = {i: word for i, word in enumerate(opinion_words)}
        opinion_word2key = {word: i for i, word in key2opinion_word.iteritems()}

        # state representation as matrix (i: word, j sentence)
        M = np.zeros((len(targets), len(opinion_words)))
        S = np.zeros((len(targets),))
        I = np.identity(len(targets))
        
        # addding data to M
        for key, value in assoc_probs.items():
            target = key[0]
            opinion_word = key[1]
            index_target = target2key[target]
            index_opinion_word = opinion_word2key[opinion_word]
            M[index_target, index_opinion_word] = value
        
        # addding data to C
        for key, value in importance.items():
            target = key
            index_target = target2key[target]
            S[index_target,] = value
        
        B = (1 - lamb) * M.dot(np.transpose(M)) 
        
        tsum = I
        for i in range(1, 101):
            tsum += LA.matrix_power(B, i)
        
        C = lamb*S.dot(tsum)
        
        confidence = {value: C[key] for key, value in key2target.items()}
        
        sorted_confidence = sorted(confidence.items(), key=lambda x: x[1], reverse=True)
    
        return sorted_confidence

# ------- FUNCTIONS --------------------------------------------------------------------------------------------------


def assoc_probs_alignment_matrix(aligner, sentences, key=lambda x: x.string):
        
        aligner.estimate(sentences, sentences)
    
        # tuple order : NN, JJ
        pairs = defaultdict(int) 
        alignment_matrix = defaultdict(set)
        
        for sentence in sentences:
            aligned_tokens = aligner.align(sentence, sentence)
    
            for token, aligned_token in zip(sentence, aligned_tokens): 
                if null not in [token, aligned_token]:
                    if token.pos.startswith('NN'):
                        if aligned_token.pos.startswith('JJ'):    
                            pairs[key(token), key(aligned_token)] += 1.0
                            alignment_matrix[key(token), key(aligned_token)].add(sentence.id) 
                        else:
                            # print token, aligned_token
                            pass
                    if token.pos.startswith('JJ'):
                        if aligned_token.pos.startswith('NN'):
                            pairs[key(aligned_token), key(token)] += 1.0
                            alignment_matrix[key(aligned_token), key(token)].add(sentence.id)
                        else:
                            # print token, aligned_token
                            pass

        nn_counts, jj_counts = single_counts(pairs)
        assoc_probs = association_probs(pairs, nn_counts, jj_counts)
        
        return assoc_probs, alignment_matrix
    

def group_sentences(sentences, terms):
    """
    Re-tokenizes sentences based on the provided multi-word terms.
    Whenever several terms appear on a sentence, the ones whose positions
    are contained into others are skipped, so finally only terms that have
    no super sets in a sentence are grouped. Chunks, relations and parse trees
    are lost during re-tokenization. 
    
    Input:
        - sentences    : list of sentences
        - terms        :  list of terms

    Output:
        - list of re-tokenized sentences
    
    """
    
    def includes(value_1, value_2):
        a_1, b_1 = value_1
        a_2, b_2 = value_2
        if a_2 >= a_1 and b_2 <= b_1:
            return True
        return False
    
    new_sentences = []
    for sentence in sentences:
        sentence_terms = []
        # find term in sentence
        for term in terms:
            value = sentence.contains(term.split())
            if value: 
                sentence_terms.append((term, value))
        # if there are terms in the sentence, make sure to keep only 
        # the ones that have no supersets
        if len(sentence_terms) > 1: 
            for index, (sterm, svalue) in enumerate(sentence_terms):
                if any([includes(svalue, o_value) for o_sterm, o_value
                        in sentence_terms if o_sterm != sterm]):
                    sentence_terms.pop(index)
            sentence_terms = sorted(sentence_terms, key=lambda x: x[1])
        if sentence_terms:
            # build new "empty" sentence
            new_sentence = Sentence(string=sentence.string, id=sentence.id)
            last_index = 0
            for sentence_term, value in sentence_terms:
                start, end = value
                # add the tokens that appear after before the previous group
                # and the beginning of the current one
                for i in range(last_index, start):
                    token_i = sentence[i]
                    new_sentence.append(string=token_i.string, pos_tag=token_i.pos_tag, lemma=token_i.lemma)
                # add the group as a single token, with NN pos_tag
                string = " ".join([token.string for token in sentence[start:end]])
                lemma = " ".join([token.lemma for token in sentence[start:end]])
                new_sentence.append(string=string, pos_tag="NN", lemma=lemma)
                last_index = end+1
            # finish with the section from last_index to the end of the sentence
            if last_index < len(sentence):
                for i in range(last_index, len(sentence)):
                    token_i = sentence[i]
                    new_sentence.append(string=token_i.string, pos_tag=token_i.pos_tag, lemma=token_i.lemma)
        else:
            new_sentence = sentence
        new_sentences.append(new_sentence)
    return new_sentences


def simple_group_nps(sentences):
    """
    Re tokenizes the sentences by grouping NPs with a simple strategy,
    Chunks and other sentence properties are lost"""
    new_sentences = []
    for sentence in sentences:
        new_sentence = Sentence(string=sentence.string, id=sentence.id)
        nps = []
        for token in sentence:
            if nps:
                if token.pos.startswith('NN') and token.chunk and token.chunk.pos == 'NP':
                    nps += [token] 
                else:
                    string = ' '.join([np.string for np in nps])
                    if sentence.is_lemmatized:
                        lemma = ' '.join([np.lemma for np in nps])
                    else:
                        lemma = None
                    new_sentence.append(string=string, lemma=lemma, pos_tag='NN')
                    new_sentence.append(string=token.string, lemma=token.lemma, pos_tag=token.pos)
                    nps = []
            else:
                if token.pos.startswith('NN') and token.chunk and token.chunk.pos == 'NP':
                    nps += [token]
                else:
                    new_sentence.append(string=token.string, lemma=token.lemma, pos_tag=token.pos)
        new_sentences.append(new_sentence)
    return new_sentences


def calc_frequency(word, sentences):
    counter = 0
    for sentence in sentences:
        if sentence.contains(word):
            counter += 1
    return counter


def single_counts(pairs):
    nn_counts = defaultdict(int)
    jj_counts = defaultdict(int)
    for (nn, jj), count in pairs.items():
        nn_counts[nn] += count
        jj_counts[jj] += count
    return nn_counts, jj_counts


def association_probs(pairs, nn_counts, jj_counts, t=0.5):
    assoc_probs = dict()
    for (nn, jj), pair_count in pairs.items():
        nn_jj_prob = pair_count / nn_counts[nn]
        jj_nn_prob = pair_count / jj_counts[jj]
        assoc_probs[(nn, jj)] = 1.0/((t/nn_jj_prob) + ((1-t)/jj_nn_prob))
    return assoc_probs  
