#!/usr/bin/python
# -*- coding: utf-8 -*-

from .token import Token
from .chunk import Chunk
from .tree import Tree
from .dependency import Relation
from .utils import _uid, BEGIN, OUTSIDE

# --- SENTENCE-----------------------------------------------------------------


class Sentence(object):

    def __init__(self, string="", id=None, document=None, parent=None):
        """
        TO DO
        """
        # Convert string to Unicode.
        if not isinstance(string, unicode):
            for enc in (("utf-8",), ("windows-1252",), ("utf-8", "ignore")):
                try:
                    string = string.decode(*enc)
                except:
                    print "Could not unicode the input"

        self.string = string
        self.parent = parent if parent else None
        self._document = document
        self.id = id if id else _uid()
        self.tokens = []
        self.chunks = []
        self.root = Token(self, "ROOT", lemma="ROOT", index=None)
        self._tree = None
        self.relations = []
        self.pipeline = []

        # helper for adding relations
        self._pending_rels = []

    @property
    def document(self):
        return self._document

    review = document

    @property
    def lemmata(self):
        return [token.lemma for token in self.tokens]

    lemma = lemmata

    @property
    def pos_tags(self):
        return [token.pos for token in self.tokens]

    parts_of_speech = pos_tags

    def _set_tree(self, tree_string, tree_strings=[]):
        """
        "mix" trees in case multiple are given

        Add extra rules to fix problems with this kind of trees returned by
        the Stanford Parser (note the empty sub trees):

        (S (SBAR (IN Although) (S (NP (NN GALAXY) (NN S5)) (VP (VBZ is) (NP (NP (DT the)
        (JJR bigger) (NN size)) (NP (NP (CD 5.1)) ('' '') (PP (IN than) (NP (NNP iPhone) (NNS 5s))))))))
        (VP (VBP () (NP (NP (CD 4)) ('' '') (NP (NP (NP (NN ))) (PRN (, ,) (S (NP (PRP it)) (VP (VBZ is)
        (NP (NP (DT the) (NN mainstream) (NN size)) (PP (IN of) (NP (NNP Android) (NN smartphone)))))) (, ,)))
        (PP (IN for) (NP (NN example)))) (, ,) (NP (NNP Xperia) (NN Z2)))) (. .))

        """
        def fix_senna_malformed(tree_string):
            """
            dealing with malformed Senna Trees due to -RBR-
            dealinf with Stanford malformed Trees due to ( and )

            Ex1:
            the player usually plays dvd 's , but has occasional problems such as :
            1 ) not recognizing a dvd
            2 ) stopping a particular point in a movie every time we played it
            3 ) not being able to access certain special features on a rental dvd .

            Tree:
            (S(S(NP(DT the)(NN player))(ADVP(RB usually))(VP(VBZ plays)(NP(NN dvd)(POS 's))))(, ,)
            (CC but)(VBZ has)(NP(NP(JJ occasional)(NNS problems))(PP(JJ such)(IN as)(: :)(NP(LS 1))))(-RRB- ))
            (RB not)(VP(VBG recognizing)(NP(DT a)(NN dvd)(CD 2)))(-RRB- ))(VBG stopping)(NP(DT a)(JJ particular)
            (NN point))(PP(IN in)(NP(DT a)(NN movie)))(NP(NP(DT every)(NN time))(SBAR(S(NP(PRP we))(VP(VBD played)
            (NP(PRP it))(NP(CD 3))))))(-RRB- ))(RB not)(VP(VBG being)(ADJP(JJ able)(S(VP(TO to)(VP(VB access)
            (NP(NP(JJ certain)(JJ special)(NNS features))(PP(IN on)(NP(DT a)(JJ rental)(NN dvd)))))))))(. .))
            """
            lefts = tree_string.count("(")
            rights = tree_string.count(")")
            total = lefts-rights
            if total:
                # more (s than )s
                if total > 0:
                    for i in range(abs(total)):
                        tree_string += ")"
                # less (s than )s
                else:
                    # if "-RRB-" in tree_string:
                    #    tree_string = tree_string.replace(")(-RRB-", "(-RRB-")
                    # else:
                    # dirty, but works
                    for i in range(abs(total)):
                        tree_string = "(" + tree_string
            return tree_string

        if tree_strings:
            trees = []
            for tree_string in tree_strings:
                try:
                    tree = Tree.fromstring(tree_string)
                except ValueError as e:
                    tree_string = fix_senna_malformed(tree_string)
                    tree = Tree.fromstring(tree_string)
                finally:
                    trees.append(tree)

            # we add the first S Tree of each clause to a bigger S Tree
            tree = Tree('S', [Tree('S', [tree[(0,)] for tree in trees])])

        else:
            try:
                tree = Tree.fromstring(tree_string)
            except ValueError as e:
                tree_string = fix_senna_malformed(tree_string)
                tree = Tree.fromstring(tree_string)

        if self.is_tokenized \
        and self.is_tagged:
            self._tree_string = tree_string
            # change the pointer of the final subtrees/leaves to Token objects
            positions = []
            for position in tree.treepositions():
                st = tree[position]
                # if isinstance(st, Tree) and st.height() <= 2:
                if isinstance(st, Tree) \
                and st.height() == 2 \
                and st.label() \
                and not isinstance(st[0], Tree):
                    positions.append(position)

            for index, position in enumerate(positions):
                tree[position] = self.tokens[index]

            self._tree = tree
        else:
            raise Exception("Sentence needs to be tokenized \
                            and pos-tagged to set its tree")

    def _get_tree(self):
        return self._tree

    tree = syntax_tree = property(_get_tree, _set_tree)

    @property
    def tagged(self):
        return [(token.string, token.pos) for token in self]

    @property
    def start(self):
        return 0

    @property
    def stop(self):
        return self.start + len(self.tokens)

    @property
    def nouns(self):
        return [token for token in self if token.pos.startswith("NN")]

    @property
    def verbs(self):
        return [token for token in self if token.pos.startswith("VB")]

    @property
    def adjectives(self):
        return [token for token in self if token.pos.startswith("JJ")]

    @property
    def nps(self):
        if self.is_chunked:
            return [chunk for chunk in self.chunks if chunk.type == 'NP']
        return []

    @property
    def parse_nps(self):
        if self.is_parsed:
            return [subtree for subtree
                    in self.tree.subtrees(lambda st: st.label() == 'NP')]
        return []

    @property
    def is_question(self):
        return len(self) > 0 and str(self[-1]) == "?"

    @property
    def is_exclamation(self):
        return len(self) > 0 and str(self[-1]) == "!"

    def __getitem__(self, index):
        return self.tokens[index]

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return self.tokens.__iter__()

    def contains(self, words, key=lambda x: x.string.lower()):
        """
        Check if word/phrase is inside the sentence.

        Input:
        words: string or iterable of words
        key  : function to be applied to every token in the sentence
               by default we use the lowercase of each token.string.

        Output
        If True, returns the first position of the word, otherwise
        returns None.
        """

        big = [key(token) for token in self.tokens]

        if isinstance(words, basestring):
            small = [words]
        else:
            small = words

        if len(small) == 1:
            small = small[0]
            if small in big:
                return big.index(small), big.index(small)
            return None
        else:
            for i in xrange(len(big)-len(small)+1):
                for j in xrange(len(small)):
                    if small[j] is None:
                        pass
                    elif big[i+j] != small[j]:
                        break
                else:
                    return i, i+len(small)
            return None

    @property
    def is_tokenized(self):
        return True if len(self.tokens) > 0 else False

    @property
    def is_lemmatized(self):
        return self.is_tokenized and all([token.is_lemmatized
                                          for token in self.tokens])

    @property
    def is_tagged(self):
        return self.is_tokenized and all([token.is_tagged
                                          for token in self.tokens])

    @property
    def is_chunked(self):
        return all([token.is_chunked for token in self.tokens])

    @property
    def is_parsed(self):
        return True if self.tree else False

    @property
    def is_dep_parsed(self):
        return True if self.relations else False

    @property
    def clauses(self):
        """
        Returns non-overlapping clauses as new Sentences
        using Slice (i.e. relations that span between the two
        sentences are dropped).

        Here, sub-sentences are considered to be subtrees of the
        penn constituency parse tree with label 'S'.

        Returns an empty list if there are no sub-sentences
        inside.
        """
        clauses = [s.leaves() for s in
                   self.tree.subtrees(lambda t: t.label() == "S"
                                                and t.height() < self.tree.height()-1)]
        real_clauses = []
        # make sure  clauses are not contained in each other
        s_clauses = [self.slice(clause[0].index, clause[-1].index+1)
                     for clause in clauses]
        for s_clause in s_clauses:
            if sum([s_clause.string in iter_clause.string
                   for iter_clause in s_clauses]) == 1:
                real_clauses.append(s_clause)
        return real_clauses

    def append(self, string=None, start=None, end=None,
               lemma=None, pos_tag=None, iob_tag=None):
        """ Appends the next Token to the Sentence / Chunk."""
        self._do_token(string, start, end, lemma, pos_tag, iob_tag)
        if iob_tag:
            self._do_chunk(iob_tag)

    def append_tags(self, lemmas=None, pos_tags=None,
                    iob_tags=None, rels=None):
        if self.is_tokenized:
            if lemmas and len(lemmas) == len(self) \
            and not self.is_lemmatized:
                for i, lemma in enumerate(lemmas):
                    self.tokens[i]._lemma = lemma

            if pos_tags and len(pos_tags) == len(self) \
            and not self.is_tagged:
                for i, pos_tag in enumerate(pos_tags):
                    self.tokens[i].pos = pos_tag

            if iob_tags and len(iob_tags) == len(self) \
            and not self.is_chunked:
                for i, iob_tag in enumerate(iob_tags):
                    self.tokens[i].iob = iob_tag
                    self._do_chunk(iob_tag, i)

            if rels:
                for head, label, dep in rels:
                    self._do_rel(head, label, dep)

    def _do_token(self, string, start, end, lemma, pos_tag, iob_tag):
        """
        Adds a new Token to the sentence.
        """
        self.tokens.append(Token(self, string, start, end,
                                 lemma, pos_tag, iob_tag,
                                 index=len(self.tokens)))

    def _do_chunk(self, iob_tag, index=None):
        """
        Adds a new Chunk to the sentence, or adds the last word to the
        previous chunk.
        The word is attached to the previous chunk if both type and
        relation match, and if the word's chunk tag does not start
        with "B-" (i.e., iob != BEGIN).
        Punctuation marks (or other "O" chunk tags) are not chunked.
        """
        tokens = self.tokens[:index+1] if index is not None else self.tokens

        if iob_tag == OUTSIDE:
            iob = OUTSIDE
            chunk_tag = None
        else:
            iob, chunk_tag = iob_tag.split('-')

        if iob != BEGIN \
        and self.chunks \
        and self.chunks[-1].type == chunk_tag \
        and tokens[-2].chunk is not None:
            self.chunks[-1].append(tokens[-1])
        else:
            ch = Chunk(self, [tokens[-1]], chunk_tag)
            self.chunks.append(ch)

    def _do_rel(self, head_index, label, dep_index):
        """
        HEAD: Head of the current token, which is either a value
        of token.index or None for ROOT.
        """
        tokens = self.tokens

        if head_index >= len(tokens):
            _pending_rel = (head_index, label, dep_index)
            if _pending_rel not in self._pending_rels:
                self._pending_rels.append(_pending_rel)
        else:
            if head_index is -1:
                r = Relation(self.root,
                             label,
                             self.tokens[dep_index])
            else:
                r = Relation(self.tokens[head_index],
                             label,
                             self.tokens[dep_index])

            self.relations.append(r)

            for i, _pending_rel in enumerate(self._pending_rels):
                p_head_index, p_label, p_dep_index = _pending_rel
                if p_head_index == self.tokens[-1].index:
                    self._do_rel(p_head_index, p_label, p_dep_index)
                    self._pending_rels.pop(i)

    def slice(self, start, stop):
        """
        Returns a portion of the sentence from word start index to word stop index.
        The returned slice is a subclass of Sentence and a deep copy.
        """
        if not self.is_tokenized:
            raise TypeError("Sentence is not tokenized")

        s = Slice()

        for i, token in enumerate(self.tokens[start:stop]):
            # The easiest way to copy (part of) a sentence
            # is by unpacking all of the token tags and passing 
            # them to Sentence.append().
            string = token.string
            lemma = token.lemma if self.is_lemmatized else None
            pos_tag = token.pos if self.is_tagged else None

            if self.is_chunked:
                # token the word belongs to is compl. included in the span
                iob_tag = token.iob
                if token.chunk.start >= start \
                and token.chunk.stop < stop:
                    s.append(string=string, lemma=lemma,
                             pos_tag=pos_tag, iob_tag=iob_tag)
                else:
                    s.append(string=string, lemma=lemma,
                             pos_tag=pos_tag, iob_tag=OUTSIDE)
            else:
                s.append(string=string, lemma=lemma, pos_tag=pos_tag)

        if self.is_parsed:
            for rel in self.relations:
                head = rel.head.index
                label = rel.label
                dependent = rel.dependent.index
                if start <= head < stop \
                and start <= dependent < stop:
                    s._do_rel(head-start, label, dependent-start)

        s.parent = self
        s.string = " ".join([token.string for token in s])
        s._start = start

        return s

    def copy(self):
        """
        Easiest way to copy is to use Slice
        """
        return self.slice(0, len(self))

    def __unicode__(self):
        return self.string

    def __repr__(self):
        if self.is_tokenized:
            if self.is_tagged:
                return "Sentence(%s)" % repr(" ".join([ "/".join([token.string, token.pos])
                                                       for token in self.tokens])).encode("utf-8")
            else:
                return "Sentence(%s)" % " ".join([token.string for
                                                  token in self.tokens]).encode("utf-8")
        else:
            return "Sentence(%s)" % self.string.encode("utf-8")

    def __eq__(self, other):
        if not isinstance(other, Sentence):
            return False
        return len(self) == len(other) and repr(self) == repr(other)


class Slice(Sentence):

    def __init__(self, *args, **kwargs):
        """ A portion of the sentence returned by Sentence.slice().
        """
        super(Slice, self).__init__()
        self._start = kwargs.pop("start", 0)

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._start + len(self.tokens)
