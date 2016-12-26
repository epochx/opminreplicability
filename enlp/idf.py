#!/usr/bin/python
# -*- coding: utf-8 -*-

from .settings import CORPORA_PATH
from os import path
from ast import literal_eval
from pandas import DataFrame
import requests
import re
import urllib2
from lxml import html
from threading import Thread
import math
import warnings
import time

"""
USAGE:

from youtubean.freq.idf import GoogleBooks
google_idf_calculator = GoogleBooks()
print google_idf_calculator.idf("Samsung Galaxy")

from youtubean.freq.idf import Wikipedia 
wiki_idf_calculator = Wikipedia()
print wiki_idf_calculator.idf("Samsung Galaxy S5")
"""


class idF(object):
    """
    Base Class for external idF calculators
    """

    def __init__(self):
        raise NotImplementedError("Needs to be implemented")

    def idf(self, query):
        """
        Calculate the idf of query (string). To input a multi-word term,
        use a whitespace-separated string
        """
        raise NotImplementedError("Needs to be implemented")

    def idfs(self, queries):
        """
        Calculates the idf of a list term. To input multi-word terms,
        use whitespace-separated strings
        This functions may use threading to improve speed.

        Input:
            - queries: list of strings to query

        Output:
            - list of idf values (positions relative to input)
        """
        raise NotImplementedError("Needs to be implemented")


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]


class GoogleBooks(idF):
    """
    idF calculator using Google Books N-gram

    What the y-axis shows is this: of all the bigrams contained in our sample of books written
    in English and published in the United States, what percentage of them are "nursery school"
    or "child care"?

    """
    MIN_COUNT = 206272

    corpora = dict(eng_us_2012=17, eng_us_2009=5, eng_gb_2012=18,
                   eng_gb_2009=6, chi_sim_2012=23, chi_sim_2009=11,
                   eng_2012=15, eng_2009=0, eng_fiction_2012=16,
                   eng_fiction_2009=4, eng_1m_2009=1, fre_2012=19, fre_2009=7,
                   ger_2012=20, ger_2009=8, heb_2012=24, heb_2009=9,
                   spa_2012=21, spa_2009=10, rus_2012=25, rus_2009=12,
                   ita_2012=22)

    def __init__(self):
        try:
            f = open(path.join(CORPORA_PATH, "googlebooks.txt"), "r")
            self.history = {k: float(v) for line in f.readlines()
                            for (k, v) in (line.strip().split("\t", 1),)}
            f.close()
        except IOError:
            self.history = {}
        self.file = path.join(CORPORA_PATH, "googlebooks.txt")

    def idfs(self, queries, write=True):
        return self.idf(",".join(queries), write=write)

    def idf(self, query, write=True):
        words = query.split(",")
        if len(words) == 1:
            word = words[0]
            value = self.history.get(word, None)
            if value is not None:
                return value
            result = self._run_query(query)
            if len(result) == 0:
                return math.log(self.MIN_COUNT)
            value = result[word][1]
            value = math.log(1.0/value)
            self.history[word] = value
            if write:
                with open(self.file, "a") as f:
                    f.write(word + "\t" + str(value) + "\n")
            return value
        else:
            final = []
            to_write = []
            for chunk in chunks(words, 10):
                result = self._run_query(",".join(chunk))
                if len(result) == 0:
                    final += [0]*len(chunk)
                else:
                    for word in chunk:
                        value = self.history.get(word, None)
                        if value is not None:
                            final.append(value)
                        else:
                            row = result.get(word, None)
                            if row is not None:
                                value = row[1]
                                value = math.log(1.0/value)
                                self.history[word] = value
                                to_write.append((word, value))
                                final.append(value)
                            else:
                                #warnings.warn(word + " not found")
                                final.append(math.log(self.MIN_COUNT))
            if write:
                with open(self.file, "a") as f:
                    for word, value in to_write:
                        f.write(word + "\t" + str(value) + "\n")
            return final

    def _get_ngrams(self, query, corpus, startYear, endYear, smoothing, caseInsensitive):
        params = dict(content=query, year_start=startYear, year_end=endYear,
                      corpus=self.corpora[corpus], smoothing=smoothing,
                      case_insensitive=caseInsensitive)
        if params['case_insensitive'] is False:
            params.pop('case_insensitive')
        if '?' in params['content']:
            params['content'] = params['content'].replace('?', '*')
        if '@' in params['content']:
            params['content'] = params['content'].replace('@', '=>')

        req = requests.get('http://books.google.com/ngrams/graph', params=params)
        while req.status_code != 200:
            warnings.warn("Received error " + str(req.status_code))
            # wait 5 minute and try again
            time.sleep(60*5)
            req = requests.get('http://books.google.com/ngrams/graph', params=params)

        res = re.findall('var data = (.*?);\\n', req.text)
        if res:
            data = {qry['ngram']: qry['timeseries']
                    for qry in literal_eval(res[0])}
            df = DataFrame(data)
            df.insert(0, 'year', list(range(startYear, endYear + 1)))
        else:
            df = DataFrame()
        return req.url, params['content'], df

    def _run_query(self, query, **params):
        """
        return
        """
        if '?' in query:
            query = query.replace('?', '*')
        if '@' in query:
            query = query.replace('@', '=>')

        # parsing params
        corpus = params.get('corpus', "eng_2012")
        startYear = params.get('startYear', 2007)
        endYear = params.get('endYear', 2008)
        smoothing = params.get('smoothing', 1)
        caseInsensitive = params.get('caseInsensitive', False)
        allData = params.get('allData', False)

        if '*' in query and caseInsensitive is True:
            caseInsensitive = False
            notifyUser = True
            warningMessage = "*NOTE: Wildcard and case-insensitive " + \
                             "searches can't be combined, so the " + \
                             "case-insensitive option was ignored."
        elif '_INF' in query and caseInsensitive is True:
            caseInsensitive = False
            notifyUser = True
            warningMessage = "*NOTE: Inflected form and case-insensitive " + \
                             "searches can't be combined, so the " + \
                             "case-insensitive option was ignored."
        else:
            notifyUser = False
        url, urlquery, df = self._get_ngrams(query, corpus, startYear, endYear,
                                             smoothing, caseInsensitive)

        if not allData:
            if caseInsensitive is True:
                for col in df.columns:
                    if col.count('(All)') == 1:
                        df[col.replace(' (All)', '')] = df.pop(col)
                    elif col.count(':chi_') == 1 or corpus.startswith('chi_'):
                        pass
                    elif col.count(':ger_') == 1 or corpus.startswith('ger_'):
                        pass
                    elif col.count(':heb_') == 1 or corpus.startswith('heb_'):
                        pass
                    elif col.count('(All)') == 0 and col != 'year':
                        if col not in urlquery.split(','):
                            df.pop(col)
            if '_INF' in query:
                for col in df.columns:
                    if '_INF' in col:
                        df.pop(col)
            if '*' in query:
                for col in df.columns:
                    if '*' in col:
                        df.pop(col)
        if notifyUser:
            print(warningMessage)

        return df


class Wikipedia(idF):
    """
    idF calculator using English Wikipedia
    """

    MIN_COUNT = 1

    def __init__(self):
        self.total_doc_count = self._get_total_articles()
        try:
            f = open(path.join(CORPORA_PATH, "wikipedia.txt"), "r")
            self.history = {k:float(v) for line in f.readlines()
                            for (k, v) in (line.strip().split("\t", 1),)}
            f.close()
        except Exception as e:
            self.history = {}
        self.file = path.join(CORPORA_PATH, "wikipedia.txt")

    def idf(self, query, write=True):
        result = self.history.get(query, None)
        if result:
            return result
        doc_count = self._get_total_results(self._http_search(query))
        if doc_count == 0:
            doc_count = self.MIN_COUNT
        value = math.log((1.0*self.total_doc_count)/doc_count)
        if query not in self.history:
            self.history[query] = value
        if write:
            with open(self.file, "a") as f:
                f.write(query + "\t" + str(value) + "\n")
        return value

    def idfs(self, queries):

        def idf_to_list(query, results, i):
            result = self.idf(query, write=False)
            results[i] = result

        results = [None] * len(queries)
        threads = []

        for i, query in enumerate(queries):
            thread = Thread(target=idf_to_list, args=(query, results, i))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        with open(self.file, "a") as f:
            for i in range(len(queries)):
                if queries[i] not in self.history:
                    self.history[queries[i]] = results[i]
                f.write(queries[i] + "\t" + str(results[i]) + "\n")

        return results

    def _http_search(self, query):
        """
        Receives an URL and requests it using urrlib2.
        Returns the received HTML as string.
        """
        query = query.replace(" ", "+")
        url = "https://en.wikipedia.org/w/index.php?title=Special%3ASearch&profile=default&search=%22" + query + "%22&fulltext=Search"
        try:
            f = urllib2.urlopen(url)
            page = f.read()
            return page
        except urllib2.HTTPError as e:
            raise e
        except urllib2.URLError as e:
            raise e

    def _get_total_results(self, results_page):
        parsed_html = html.fromstring(results_page)
        try:
            number = parsed_html.find_class("results-info")[0][1].text_content()
            return int(number.replace(",", ""))
        except Exception as e:
            #print e
            return 0

    def _get_total_articles(self):
        try:
            f = urllib2.urlopen("https://en.wikipedia.org/wiki/Main_Page")
            page = f.read()
            parsed_html = html.fromstring(page)
            return int(parsed_html.get_element_by_id("articlecount")[0].text.replace(",", ""))
        except Exception:
            return 5221542
