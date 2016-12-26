#!/usr/bin/python
# -*- coding: utf-8 -*-

from enlp.models.hu2004 import FreqAspectExtractor
from enlp.models.qiu2011 import DoublePropagation, BASIC_SEEDS
from enlp.models.liu2012 import GraphTargetExtractor
from enlp.lexicon.liu import LiuLexicon
from enlp.lexicon.manual import ManualLexicon
from enlp.models.utils import LowerStringKey, StemKey
from enlp.corpus.liu import CreativeMP3Player, ApexDVDPlayer, NikonCamera, CanonCamera, NokiaCellphone
from eval import eval_aspect_ext

import argparse
import sys
from multiprocessing import Process, Queue, cpu_count
from os import path, makedirs
import json
from collections import Mapping
from functools import partial
import operator
from itertools import product
import numpy as np


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


class ParameterGrid(object):
    """
    From sklearn
    """

    def __init__(self, param_grid):
        if isinstance(param_grid, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_grid = [param_grid]
        self.param_grid = param_grid

    def __iter__(self):
        """Iterate over the points in the grid.

        Returns
        -------
        params : iterator over dict of string to any
            Yields dictionaries mapping each estimator parameter to one of its
            allowed values.
        """
        for p in self.param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        """Number of points on the grid."""
        # Product function that can handle iterables (np.product can't).
        product = partial(reduce, operator.mul)
        return sum(product(len(v) for v in p.values()) if p else 1
                   for p in self.param_grid)

    def __getitem__(self, ind):
        # This is used to make discrete sampling without replacement memory
        # efficient.
        for sub_grid in self.param_grid:
            # XXX: could memoize information used here
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue

            # Reverse so most frequent cycling parameter comes first
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.product(sizes)

            if ind >= total:
                # Try the next grid
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                return out

        raise IndexError('ParameterGrid index out of range')


def model_results_corpus(Model, params, corpus, gold_standards):
    """
    The function run when using multiprocessing.

    :param Model:
    :param params:
    :param corpus:
    :param gold_standards:
    :return:
    """
    model = Model(**params)
    if Model == FreqAspectExtractor:
        aspects = model.extract(corpus.sentences)
        model_aspects = set()
        for aspect in aspects:
            model_aspects |= aspect.compact_aspects
    elif Model == DoublePropagation:
        aspects, opinion_words = model.extract(corpus.sentences)
        token_tuples = []
        for aspect in aspects:
            for ais in aspect.sentences:
                token_tuples += ais.tokens
        model_aspects = set([" ".join([token.string.lower() for token in token_tuple])
                             for token_tuple in token_tuples])

    elif Model == GraphTargetExtractor:
        aspects = model.extract(corpus.sentences)
        model_aspects = [aspect for aspect, confidence in aspects]

    result = eval_aspect_ext(model_aspects, gold_standards[corpus.name])

    params = params.items()

    if Model != GraphTargetExtractor:
        data = [[str(value) for name, value in params] +
                [corpus.name, corpus.pipeline] +
                [result.p, result.recall, result.fmeasure]]
    else:
        data = []
        for n in range(10, len(model_aspects), 10):
            lim_result = eval_aspect_ext(model_aspects[:n], gold_standards[corpus.name])
            datum = [str(value) for name, value in params] + \
                    [str(n)] + \
                    [corpus.name, corpus.pipeline] + \
                    [lim_result.p, lim_result.recall, lim_result.fmeasure]
            data.append(datum)
    return data


def run_model(Model, params_comb, corpora, gold_standards, func, save_path, processes=10, save_chunk=5):
    """
    Input:
        - Model         : class of the model
        - params_list   : list of parameter dicts
        - corpora       : list of corpora to apply the model
        - func          : function that runs the model and get extracted aspects
        - processes     : int, number of CPUs to use
        - sove_chunk  : int, step to pickle results
        - load          : boolean, True to continue loading on the existing pickle

    Output:
        - list of tuples (params, corpus.name, model_aspects)

    """
    def worker(input, output):
        for func, args in iter(input.get, 'STOP'):
            try:
                result = func(*args)
                output.put((True, result))
            except Exception as e:
                output.put((False, e))


    if Model != GraphTargetExtractor:
        names = [name.replace("_", " ") for name, value in params_comb.items()] + \
                ["Corpus", "Pipeline", "P", "R", "TF"]
    else:
        names = [name.replace("_", " ") for name, value in params_comb.items()] + \
                ["n"] + ["Corpus", "Pipeline", "P", "R", "TF"]

    result = dict()
    result["names"] = names
    result["data"] = []

    params_list = ParameterGrid(params_comb)

    try:
        NUMBER_OF_PROCESSES = processes

        TASKS = [(func, (Model, params, corpus, gold_standards))
                 for params in params_list for corpus in corpora]

        # Create queues
        task_queue = Queue()
        done_queue = Queue()

        progress = 0
        processes_container = []
        # Start worker processes
        for i in range(NUMBER_OF_PROCESSES):
            p = Process(target=worker, args=(task_queue, done_queue))
            processes_container.append(p)
            p.start()

        for chunk in chunks(TASKS, save_chunk):
            # Submit tasks
            for task in chunk:
                task_queue.put(task)

            # Get and print results
            for i in range(len(chunk)):
                state, output = done_queue.get()
                progress += 1.0
                if state:
                    result["data"] += output
                else:
                    print output
                print '[progress] >> %2.2f%%\r' % (progress/len(TASKS)*100.),
                sys.stdout.flush()

            with open(path.join(save_path, "{0}_results.json".format(Model.__name__)), "wb") as f:
                json.dump(result, f)

        # Tell child processes to stop
        for i in range(NUMBER_OF_PROCESSES):
            task_queue.put('STOP')

    except KeyboardInterrupt:
        print("Saving results and stopping processes...\n")

        with open(path.join(save_path, "{0}_results.json".format(Model.__name__)), "wb") as f:
            json.dump(result, f)

        for process in processes_container:
            process.terminate()

        sys.exit(0)

if __name__ == "__main__":

    desc = "Help for run, a script that runs classic opinion mining algorithms"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("save_path",
                        help="Absolute path to store JSON  result files")

    models = ["FreqAspectExtractor", "DoublePropagation", "GraphTargetExtractor"]

    parser.add_argument('--models', "-m",
                        nargs='*',
                        choices=models,
                        help="Models to run. Allowed values are " + ', '.join(models),
                        metavar='')

    Corpora = {"CreativeMP3Player": CreativeMP3Player,
               "ApexDVDPlayer": ApexDVDPlayer,
               "NikonCamera": NikonCamera,
               "CanonCamera": CanonCamera,
               "NokiaCellphone": NokiaCellphone}

    parser.add_argument('--corpus', "-c",
                        nargs='*',
                        choices=Corpora,
                        help="Names of corpus to use. Allowed values are " + ', '.join(Corpora),
                        metavar='')

    parser.add_argument('--cpu-count', "-cpu",
                        nargs='*',
                        type=int,
                        default=cpu_count(),
                        help="Number of CPU cores to use. Default is CPU count.")

    args = parser.parse_args()

    save_path = args.save_path
    if not path.isdir(save_path):
        makedirs(save_path)

    if args.models is None:
        model_names = models
    else:
        model_names = args.models

    if args.corpus is None:
        corpus_names = Corpora.keys()
    else:
        corpus_names = args.models

    processes = args.cpu_count
    if processes > cpu_count():
        processes = cpu_count()

    corpora = []
    gold_standards = {}

    for corpus_name in corpus_names:
        Corpus = Corpora[corpus_name]
        corpus = Corpus()
        gold_standard = [aspect.string for aspect in corpus.aspects]
        gold_standards[corpus.name] = gold_standard
        for pipeline in Corpus.list_frozen():
            corpus = Corpus.unfreeze(pipeline)
            corpora.append(corpus)

    if "FreqAspectExtractor" in model_names:

        hu2004_params_comb = {"min_support": [0.2, 0.6, 0.8, 1, 1.2, 1.4],
                              "max_words": [2, 3, 4],
                              "min_psupport": [1, 2, 3],
                              "min_compact_support": [1, 2, 3],
                              "max_compact_distance": [1, 3, 5],
                              "adj_key": [StemKey(), LowerStringKey()],
                              "adj_win_size": [2, 3, 4, 5],
                              "match": [85, 90, 95]}

        run_model(FreqAspectExtractor,
                  hu2004_params_comb,
                  corpora,
                  gold_standards,
                  model_results_corpus,
                  save_path,
                  processes=processes,
                  save_chunk=processes)

    if "DoublePropagation" in model_names:

        lexicon = LiuLexicon()
        lexicon_chunks = []

        for i, chunk in enumerate(chunks(lexicon.items(), int(len(lexicon.items()) / 9)), 1):
            lexicon_chunk = ManualLexicon(lexicon.name + " " + str(i))
            for key, value in chunk:
                lexicon_chunk[key] = value
            lexicon_chunks.append(lexicon_chunk)

        qiu2011_params_comb = {"seeds": [BASIC_SEEDS] + lexicon_chunks,
                               "negation_limit": [3, 5],
                               "Q": [1, 2],
                               "K": [3, 4],
                               "key": [LowerStringKey()],
                               "match": [80, 90, 100]}

        run_model(DoublePropagation,
                  qiu2011_params_comb,
                  corpora,
                  gold_standards,
                  model_results_corpus,
                  save_path,
                  processes=processes,
                  save_chunk=processes)

    if "GraphTargetExtractor" in model_names:

        liu2012_params_comb = {"grouping": ["ngram", "simple", "condition"],
                               "limit": [50, 100],
                               "key": [LowerStringKey(), StemKey()],
                               "idf": ["wikipedia"]}

        run_model(GraphTargetExtractor,
                  liu2012_params_comb,
                  corpora,
                  gold_standards,
                  model_results_corpus,
                  save_path,
                  processes=processes,
                  save_chunk=processes)
