#!/usr/bin/python
# -*- coding: utf-8 -*-


"""
 Various Tree objects for representing syntax or morphological trees
 This is an adaptation of the original tree.py file from NLTK.
 Original notice:
 Natural Language Toolkit: Text Trees

 Copyright (C) 2001-2010 NLTK Project
 Author: Edward Loper <edloper@gradient.cis.upenn.edu>
         Steven Bird <sb@csse.unimelb.edu.au>
         Nathan Bodenstab <bodenstab@cslu.ogi.edu> (tree transforms)
 URL: <http://www.nltk.org/>
 For license information, see LICENSE.TXT
"""

import re
import sys

if sys.version_info[0] == 2:
    from cgi import escape as htmlescape
else:
    from html import escape as htmlescape
    unicode = str

string_types = basestring,


def unicode_repr(obj):
    """
    For classes that was fixed with @python_2_unicode_compatible
    ``unicode_repr`` returns ``obj.unicode_repr()``; for unicode strings
    the result is returned without "u" letter (to make output the
    same under Python 2.x and Python 3.x); for other variables
    it is the same as ``repr``.
    """
    # return repr(obj)[1:] # strip "u" letter from output
    return repr(obj)


class Tree(list):
    """
    A Tree represents a hierarchical grouping of leaves and subtrees.
    For example, each constituent in a syntax tree is represented by a single Tree.

    A tree's children are encoded as a list of leaves and subtrees,
    where a leaf is a basic (non-tree) value; and a subtree is a
    nested Tree.

        >>> from nltk.tree import Tree
        >>> print(Tree(1, [2, Tree(3, [4]), 5]))
        (1 2 (3 4) 5)
        >>> vp = Tree('VP', [Tree('V', ['saw']),
        ...                  Tree('NP', ['him'])])
        >>> s = Tree('S', [Tree('NP', ['I']), vp])
        >>> print(s)
        (S (NP I) (VP (V saw) (NP him)))
        >>> print(s[1])
        (VP (V saw) (NP him))
        >>> print(s[1,1])
        (NP him)
        >>>
        >>> s == t
        True
        >>> t[1][1].set_label('X')
        >>> t[1][1].label()
        'X'
        >>> print(t)
        (S (NP I) (VP (V saw) (X him)))
        >>> t[0], t[1,1] = t[1,1], t[0]
        >>> print(t)
        (S (X him) (VP (V saw) (NP I)))

    The length of a tree is the number of children it has.

        >>> len(t)
        2

    The set_label() and label() methods allow individual constituents
    to be labeled.  For example, syntax trees use this label to specify
    phrase tags, such as "NP" and "VP".

    Several Tree methods use "tree positions" to specify
    children or descendants of a tree.  Tree positions are defined as
    follows:

      - The tree position *i* specifies a Tree's *i*\ th child.
      - The tree position ``()`` specifies the Tree itself.
      - If *p* is the tree position of descendant *d*, then
        *p+i* specifies the *i*\ th child of *d*.

    I.e., every tree position is either a single index *i*,
    specifying ``tree[i]``; or a sequence *i1, i2, ..., iN*,
    specifying ``tree[i1][i2]...[iN]``.

    Construct a new tree.  This constructor can be called in one
    of two ways:

    - ``Tree(label, children)`` constructs a new tree with the
        specified label and list of children.

    - ``Tree.fromstring(s)`` constructs a new tree by parsing the string ``s``.
    """
    def __init__(self, node, children=None):
        if children is None:
            raise TypeError("%s: Expected a node value and child list "
                                % type(self).__name__)
        elif isinstance(children, string_types):
            raise TypeError("%s() argument 2 should be a list, not a "
                            "string" % type(self).__name__)
        else:
            list.__init__(self, children)
            self._label = node

    #////////////////////////////////////////////////////////////
    # Comparison operators
    #////////////////////////////////////////////////////////////

    def __eq__(self, other):
        return (self.__class__ is other.__class__ and
                (self._label, list(self)) == (other._label, list(other)))

    def __lt__(self, other):
        if not isinstance(other, Tree):
            # raise_unorderable_types("<", self, other)
            # Sometimes children can be pure strings,
            # so we need to be able to compare with non-trees:
            return self.__class__.__name__ < other.__class__.__name__
        elif self.__class__ is other.__class__:
            return (self._label, list(self)) < (other._label, list(other))
        else:
            return self.__class__.__name__ < other.__class__.__name__

    # @total_ordering doesn't work here, since the class inherits from a builtin class
    __ne__ = lambda self, other: not self == other
    __gt__ = lambda self, other: not (self < other or self == other)
    __le__ = lambda self, other: self < other or self == other
    __ge__ = lambda self, other: not self < other

    #////////////////////////////////////////////////////////////
    # Disabled list operations
    #////////////////////////////////////////////////////////////

    def __mul__(self, v):
        raise TypeError('Tree does not support multiplication')

    def __rmul__(self, v):
        raise TypeError('Tree does not support multiplication')

    def __add__(self, v):
        raise TypeError('Tree does not support addition')

    def __radd__(self, v):
        raise TypeError('Tree does not support addition')

    #////////////////////////////////////////////////////////////
    # Indexing (with support for tree positions)
    #////////////////////////////////////////////////////////////

    def __getitem__(self, index):
        if isinstance(index, (int, slice)):
            return list.__getitem__(self, index)
        elif isinstance(index, (list, tuple)):
            if len(index) == 0:
                return self
            elif len(index) == 1:
                return self[index[0]]
            else:
                return self[index[0]][index[1:]]
        else:
            raise TypeError("%s indices must be integers, not %s" %
                            (type(self).__name__, type(index).__name__))

    def __setitem__(self, index, value):
        if isinstance(index, (int, slice)):
            return list.__setitem__(self, index, value)
        elif isinstance(index, (list, tuple)):
            if len(index) == 0:
                raise IndexError('The tree position () may not be '
                                 'assigned to.')
            elif len(index) == 1:
                self[index[0]] = value
            else:
                self[index[0]][index[1:]] = value
        else:
            raise TypeError("%s indices must be integers, not %s" %
                            (type(self).__name__, type(index).__name__))

    def __delitem__(self, index):
        if isinstance(index, (int, slice)):
            return list.__delitem__(self, index)
        elif isinstance(index, (list, tuple)):
            if len(index) == 0:
                raise IndexError('The tree position () may not be deleted.')
            elif len(index) == 1:
                del self[index[0]]
            else:
                del self[index[0]][index[1:]]
        else:
            raise TypeError("%s indices must be integers, not %s" %
                            (type(self).__name__, type(index).__name__))

    #////////////////////////////////////////////////////////////
    # Basic tree operations
    #////////////////////////////////////////////////////////////

    def _get_node(self):
        """Outdated method to access the node value; use the label() method instead."""
        raise NotImplementedError("Use label() to access a node label.")

    def _set_node(self, value):
        """Outdated method to set the node value; use the set_label() method instead."""
        raise NotImplementedError("Use set_label() method to set a node label.")
    node = property(_get_node, _set_node)

    def label(self):
        """
        Return the node label of the tree.

            >>> t = Tree.fromstring('(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))')
            >>> t.label()
            'S'

        :return: the node label (typically a string)
        :rtype: any
        """
        return self._label

    def set_label(self, label):
        """
        Set the node label of the tree.

            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
            >>> t.set_label("T")
            >>> print(t)
            (T (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))

        :param label: the node label (typically a string)
        :type label: any
        """
        self._label = label

    def leaves(self):
        """
        Return the leaves of the tree.

            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
            >>> t.leaves()
            ['the', 'dog', 'chased', 'the', 'cat']

        :return: a list containing this tree's leaves.
            The order reflects the order of the
            leaves in the tree's hierarchical structure.
        :rtype: list
        """
        leaves = []
        for child in self:
            if isinstance(child, Tree):
                leaves.extend(child.leaves())
            else:
                leaves.append(child)
        return leaves

    def flatten(self):
        """
        Return a flat version of the tree, with all non-root non-terminals removed.

            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
            >>> print(t.flatten())
            (S the dog chased the cat)

        :return: a tree consisting of this tree's root connected directly to
            its leaves, omitting all intervening non-terminal nodes.
        :rtype: Tree
        """
        return Tree(self.label(), self.leaves())

    def height(self):
        """
        Return the height of the tree.

            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
            >>> t.height()
            5
            >>> print(t[0,0])
            (D the)
            >>> t[0,0].height()
            2

        :return: The height of this tree.  The height of a tree
            containing no children is 1; the height of a tree
            containing only leaves is 2; and the height of any other
            tree is one plus the maximum of its children's
            heights.
        :rtype: int
        """
        max_child_height = 0
        for child in self:
            if isinstance(child, Tree):
                max_child_height = max(max_child_height, child.height())
            else:
                max_child_height = max(max_child_height, 1)
        return 1 + max_child_height

    def treepositions(self, order='preorder'):
        """
            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
            >>> t.treepositions() # doctest: +ELLIPSIS
            [(), (0,), (0, 0), (0, 0, 0), (0, 1), (0, 1, 0), (1,), (1, 0), (1, 0, 0), ...]
            >>> for pos in t.treepositions('leaves'):
            ...     t[pos] = t[pos][::-1].upper()
            >>> print(t)
            (S (NP (D EHT) (N GOD)) (VP (V DESAHC) (NP (D EHT) (N TAC))))

        :param order: One of: ``preorder``, ``postorder``, ``bothorder``,
            ``leaves``.
        """
        positions = []
        if order in ('preorder', 'bothorder'): positions.append(())
        for i, child in enumerate(self):
            if isinstance(child, Tree):
                childpos = child.treepositions(order)
                positions.extend((i,)+p for p in childpos)
            else:
                positions.append((i,))
        if order in ('postorder', 'bothorder'): positions.append(())
        return positions

    def subtrees(self, filter=None):
        """
        Generate all the subtrees of this tree, optionally restricted
        to trees matching the filter function.

            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
            >>> for s in t.subtrees(lambda t: t.height() == 2):
            ...     print(s)
            (D the)
            (N dog)
            (V chased)
            (D the)
            (N cat)

        :type filter: function
        :param filter: the function to filter all local trees
        """
        if not filter or filter(self):
            yield self
        for child in self:
            if isinstance(child, Tree):
                for subtree in child.subtrees(filter):
                    yield subtree

    def pos(self):
        """
        Return a sequence of pos-tagged words extracted from the tree.

            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
            >>> t.pos()
            [('the', 'D'), ('dog', 'N'), ('chased', 'V'), ('the', 'D'), ('cat', 'N')]

        :return: a list of tuples containing leaves and pre-terminals (part-of-speech tags).
            The order reflects the order of the leaves in the tree's hierarchical structure.
        :rtype: list(tuple)
        """
        pos = []
        for child in self:
            if isinstance(child, Tree):
                pos.extend(child.pos())
            else:
                pos.append((child, self._label))
        return pos

    def leaf_treeposition(self, index):
        """
        :return: The tree position of the ``index``-th leaf in this
            tree.  I.e., if ``tp=self.leaf_treeposition(i)``, then
            ``self[tp]==self.leaves()[i]``.

        :raise IndexError: If this tree contains fewer than ``index+1``
            leaves, or if ``index<0``.
        """
        if index < 0:
            raise IndexError('index must be non-negative')

        stack = [(self, ())]
        while stack:
            value, treepos = stack.pop()
            if not isinstance(value, Tree):
                if index == 0:
                    return treepos
                else:
                    index -= 1
            else:
                for i in range(len(value)-1, -1, -1):
                    stack.append((value[i], treepos+(i,)))

        raise IndexError('index must be less than or equal to len(self)')

    def treeposition_spanning_leaves(self, start, end):
        """
        :return: The tree position of the lowest descendant of this
            tree that dominates ``self.leaves()[start:end]``.
        :raise ValueError: if ``end <= start``
        """
        if end <= start:
            raise ValueError('end must be greater than start')
        # Find the tree positions of the start & end leaves, and
        # take the longest common subsequence.
        start_treepos = self.leaf_treeposition(start)
        end_treepos = self.leaf_treeposition(end-1)
        # Find the first index where they mismatch:
        for i in range(len(start_treepos)):
            if i == len(end_treepos) or start_treepos[i] != end_treepos[i]:
                return start_treepos[:i]
        return start_treepos


    #////////////////////////////////////////////////////////////
    # Convert, copy
    #////////////////////////////////////////////////////////////

    @classmethod
    def convert(cls, tree):
        """
        Convert a tree between different subtypes of Tree.  ``cls`` determines
        which class will be used to encode the new tree.

        :type tree: Tree
        :param tree: The tree that should be converted.
        :return: The new Tree.
        """
        if isinstance(tree, Tree):
            children = [cls.convert(child) for child in tree]
            return cls(tree._label, children)
        else:
            return tree

    def copy(self, deep=False):
        if not deep:
            return type(self)(self._label, self)
        else:
            return type(self).convert(self)

    #////////////////////////////////////////////////////////////
    # Parsing
    #////////////////////////////////////////////////////////////

    @classmethod
    def fromstring(cls, s, brackets='()', read_node=None, read_leaf=None,
                   node_pattern=None, leaf_pattern=None,
                   remove_empty_top_bracketing=False):
        """
        Read a bracketed tree string and return the resulting tree.
        Trees are represented as nested brackettings, such as::

          (S (NP (NNP John)) (VP (V runs)))

        :type s: str
        :param s: The string to read

        :type brackets: str (length=2)
        :param brackets: The bracket characters used to mark the
            beginning and end of trees and subtrees.

        :type read_node: function
        :type read_leaf: function
        :param read_node, read_leaf: If specified, these functions
            are applied to the substrings of ``s`` corresponding to
            nodes and leaves (respectively) to obtain the values for
            those nodes and leaves.  They should have the following
            signature:

               read_node(str) -> value

            For example, these functions could be used to process nodes
            and leaves whose values should be some type other than
            string (such as ``FeatStruct``).
            Note that by default, node strings and leaf strings are
            delimited by whitespace and brackets; to override this
            default, use the ``node_pattern`` and ``leaf_pattern``
            arguments.

        :type node_pattern: str
        :type leaf_pattern: str
        :param node_pattern, leaf_pattern: Regular expression patterns
            used to find node and leaf substrings in ``s``.  By
            default, both nodes patterns are defined to match any
            sequence of non-whitespace non-bracket characters.

        :type remove_empty_top_bracketing: bool
        :param remove_empty_top_bracketing: If the resulting tree has
            an empty node label, and is length one, then return its
            single child instead.  This is useful for treebank trees,
            which sometimes contain an extra level of bracketing.

        :return: A tree corresponding to the string representation ``s``.
            If this class method is called using a subclass of Tree,
            then it will return a tree of that type.
        :rtype: Tree
        """
        if not isinstance(brackets, string_types) or len(brackets) != 2:
            raise TypeError('brackets must be a length-2 string')
        if re.search('\s', brackets):
            raise TypeError('whitespace brackets not allowed')
        # Construct a regexp that will tokenize the string.
        open_b, close_b = brackets
        open_pattern, close_pattern = (re.escape(open_b), re.escape(close_b))
        if node_pattern is None:
            node_pattern = '[^\s%s%s]+' % (open_pattern, close_pattern)
        if leaf_pattern is None:
            leaf_pattern = '[^\s%s%s]+' % (open_pattern, close_pattern)
        token_re = re.compile('%s\s*(%s)?|%s|(%s)' % (
            open_pattern, node_pattern, close_pattern, leaf_pattern))
        # Walk through each token, updating a stack of trees.
        stack = [(None, [])] # list of (node, children) tuples
        for match in token_re.finditer(s):
            token = match.group()
            # Beginning of a tree/subtree
            if token[0] == open_b:
                if len(stack) == 1 and len(stack[0][1]) > 0:
                    cls._parse_error(s, match, 'end-of-string')
                label = token[1:].lstrip()
                if read_node is not None:
                    label = read_node(label)
                stack.append((label, []))
            # End of a tree/subtree
            elif token == close_b:
                if len(stack) == 1:
                    if len(stack[0][1]) == 0:
                        cls._parse_error(s, match, open_b)
                    else:
                        cls._parse_error(s, match, 'end-of-string')
                label, children = stack.pop()
                stack[-1][1].append(cls(label, children))
            # Leaf node
            else:
                if len(stack) == 1:
                    cls._parse_error(s, match, open_b)
                if read_leaf is not None:
                    token = read_leaf(token)
                stack[-1][1].append(token)

        # check that we got exactly one complete tree.
        if len(stack) > 1:
            cls._parse_error(s, 'end-of-string', close_b)
        elif len(stack[0][1]) == 0:
            cls._parse_error(s, 'end-of-string', open_b)
        else:
            assert stack[0][0] is None
            assert len(stack[0][1]) == 1
        tree = stack[0][1][0]

        # If the tree has an extra level with node='', then get rid of
        # it.  E.g.: "((S (NP ...) (VP ...)))"
        if remove_empty_top_bracketing and tree._label == '' and len(tree) == 1:
            tree = tree[0]
        # return the tree.
        return tree

    @classmethod
    def _parse_error(cls, s, match, expecting):
        """
        Display a friendly error message when parsing a tree string fails.
        :param s: The string we're parsing.
        :param match: regexp match of the problem token.
        :param expecting: what we expected to see instead.
        """
        # Construct a basic error message
        if match == 'end-of-string':
            pos, token = len(s), 'end-of-string'
        else:
            pos, token = match.start(), match.group()
        msg = '%s.read(): expected %r but got %r\n%sat index %d.' % (
            cls.__name__, expecting, token, ' '*12, pos)
        # Add a display showing the error token itsels:
        s = s.replace('\n', ' ').replace('\t', ' ')
        offset = pos
        if len(s) > pos+10:
            s = s[:pos+10]+'...'
        if pos > 10:
            s = '...'+s[pos-10:]
            offset = 13
        msg += '\n%s"%s"\n%s^' % (' '*16, s, ' '*(17+offset))
        error = ValueError(msg)
        error.index = pos
        raise error

    #////////////////////////////////////////////////////////////
    # Visualization & String Representation
    #////////////////////////////////////////////////////////////

    def __repr__(self):
        childstr = ", ".join(unicode_repr(c) for c in self)
        return '%s(%s, [%s])' % (type(self).__name__, unicode_repr(self._label), childstr)

    def _repr_png_(self):
        """
        Draws and outputs in PNG for ipython.
        PNG is used instead of PDF, since it can be displayed in the qt console and
        has wider browser support.
        """
        pass

    def __str__(self):
        return self.__repr__()


# functions to handle tree operations / to be added
# def get_lca_length(location1, location2):
#     i = 0
#     while i < len(location1) and i < len(location2) and location1[i] == location2[i]:
#         i+=1
#     return i
#
# def get_labels_from_lca(ptree, lca_len, location):
#     labels = []
#     for i in range(lca_len, len(location)):
#         labels.append(ptree[location[:i]].label())
#     return labels
#
# def find_path(ptree, token1, token2):
#     leaf_index1 = token1.index
#     leaf_index2 = token2.index
#
#     location1 = ptree.leaf_treeposition(leaf_index1)
#     location2 = ptree.leaf_treeposition(leaf_index2)
#
#     # find length of least common ancestor (lca)
#     lca_len = get_lca_length(location1, location2)
#
#     # find path from the node1 to lca
#
#     labels1 = get_labels_from_lca(ptree, lca_len, location1)
#     # ignore the first element, because it will be counted in the second part of the path
#     result = labels1[1:]
#     # inverse, because we want to go from the node to least common ancestor
#     result = result[::-1]
#
#     # add path from lca to node2
#     result = result + get_labels_from_lca(ptree, lca_len, location2)
#     return result
