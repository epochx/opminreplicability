#!/usr/bin/env python
#
""" From https://github.com/dasmith/stanford-corenlp-python
"""
import os
import re
import glob
import traceback
import pexpect
import tempfile
import shutil
import sys
import warnings
from xml.etree import ElementTree as et
from collections import namedtuple
from subprocess import call, check_output, STDOUT, CalledProcessError


from settings import CORENLP_PATH

VERBOSE = False
STATE_START, STATE_TEXT, STATE_WORDS, STATE_TREE, STATE_DEPENDENCY, STATE_COREFERENCE = 0, 1, 2, 3, 4, 5
WORD_PATTERN = re.compile('\[([^\]]+)\]')
CR_PATTERN = re.compile(r"\((\d*),(\d*),\[(\d*),(\d*)\]\) -> \((\d*),(\d*),\[(\d*),(\d*)\]\), that is: \"(.*)\" -> \"(.*)\"")

# -------------- EXCEPTIONS ---------------------------------------------------------------------------------


class ProcessError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class ParserError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class TimeoutError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class OutOfMemoryError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

# -------------- RESULTS ----------------------------------------------------

Sentence = namedtuple('Sentence', ['tokens', 'syntax_tree', 'dependencies'])
Token = namedtuple("Token", ['index', 'word', 'POS', 'lemma', 'ner',
                             'start', 'end'])
Dep = namedtuple("Dep", ["head", "label", "dependent"])
Head = namedtuple("Head", ["index", "word"])
Dependent = namedtuple("Dependent", ["index", "word"])

# -------------- MAIN CLASS --------------------------------------------------


class CoreNLP:
    """
    Command-line interaction with Stanford's CoreNLP java utilities.
    Can be run as a JSON-RPC server or imported as a module.
    """

    def __init__(self, corenlp_path=CORENLP_PATH, memory="3g", serving=False,
                 annotators=["tokenize", "ssplit", "pos", "lemma", "ner", "parse", "dcoref"], **kwargs):
        """
        Checks the location of the jar files.
        Spawns the server as a process.
        """

        # spawn the server
        self.serving = serving
        self.corenlp_path = corenlp_path
        self.memory = memory
        self.kwargs = kwargs
        self.annotators = annotators
        self.spawn_corenlp()

    def spawn_corenlp(self):
        command = generate_corenlp_command(self.corenlp_path, memory=self.memory,
                                           annotators=self.annotators, **self.kwargs)
        self.corenlp = pexpect.spawn(command, timeout=60, maxread=8192, searchwindowsize=80)
        # interactive shell
        self.corenlp.expect("\nNLP> ")

    def close(self, force=True):
        self.corenlp.terminate(force)

    def isalive(self):
        return self.corenlp.isalive()

    def __del__(self):
        # If our child process is still around, kill it
        if self.isalive():
            self.close()

    def _parse(self, text, raw_output=False):
        """
        This is the core interaction with the parser.

        It returns a Python data-structure, while the parse()
        function returns a JSON object
        """
        # CoreNLP interactive shell cannot recognize newline
        if '\n' in text or '\r' in text:
            to_send = re.sub("[\r\n]", " ", text).strip()
        else:
            to_send = text

        # clean up anything leftover
        def clean_up():
            while True:
                try:
                    self.corenlp.read_nonblocking(8192, 0.1)
                except pexpect.TIMEOUT:
                    break
        clean_up()

        self.corenlp.sendline(to_send)

        # How much time should we give the parser to parse it?
        # the idea here is that you increase the timeout as a
        # function of the text's length.
        # max_expected_time = max(5.0, 3 + len(to_send) / 5.0)
        max_expected_time = max(300.0, len(to_send) / 3.0)

        # repeated_input = self.corenlp.except("\n")  # confirm it
        t = self.corenlp.expect(["\nNLP> ", pexpect.TIMEOUT, pexpect.EOF,
                                 "\nWARNING: Parsing of sentence failed, possibly because of out of memory."],
                                timeout=max_expected_time)
        incoming = self.corenlp.before
        if t == 1:
            # TIMEOUT, clean up anything left in buffer
            clean_up()
            print >>sys.stderr, {'error': "timed out after %f seconds" % max_expected_time,
                                 'input': to_send,
                                 'output': incoming}
            raise TimeoutError("Timed out after %d seconds" % max_expected_time)
        elif t == 2:
            # EOF, probably crash CoreNLP process
            print >>sys.stderr, {'error': "CoreNLP terminates abnormally while parsing",
                                 'input': to_send,
                                 'output': incoming}
            raise ProcessError("CoreNLP process terminates abnormally while parsing")
        elif t == 3:
            # out of memory
            print >>sys.stderr, {'error': "WARNING: Parsing of sentence failed, possibly because of out of memory.",
                                 'input': to_send,
                                 'output': incoming}
            raise OutOfMemoryError

        try:
            if raw_output:
                results = incoming
            else:
                results = parse_interactive_shell_result(incoming)
        except Exception as e:
            print traceback.format_exc()
            raise e

        return results

    def parse(self, text, raw_output=False):
        """
        This function takes a text string, sends it to the Stanford parser
        interactive shell,  reads in the result, parses the results and returns 
        a list with one dictionary entry for each parsed sentence in JSON format.

        If raw_output is True, does not returns before JSON parse.
        """
        try:
            return self._parse(text, raw_output=raw_output)

        except Exception as e:
            print e  # Should probably log somewhere instead of printing
            self.corenlp.close()
            self.spawn_corenlp()
            raise e

    def batch_parse(self, texts, raw_output=False):
        """
        This function takes input files,
        sends list of input files to the Stanford parser,
        reads in the results from temporary folder in your OS and
        returns a generator object of list that consist of dictionary entry.
        If raw_output is true, the dictionary returned will correspond exactly to XML.
        ( The function needs xmltodict,
        and doesn't need init 'StanfordCoreNLP' class. )
        """

        return batch_parser(texts, raw_output=False,
                            annotators=self.annotators,
                            corenlp_path=self.corenlp_path,
                            memory=self.memory,
                            **self.kwargs)

    def tree2deps(self, tree, **options):
        input_file = tempfile.NamedTemporaryFile()

        with input_file.file as f:
            f.write(tree.encode("utf-8"))

        command = generate_tree2dep_command(corenlp_path=self.corenlp_path, memory=self.memory,
                                            tree_file=input_file.name, **options)

        # creates the xml file of parser output:
        try:
            result = check_output(command, stderr=STDOUT, shell=True)
            new_dependencies = []
            for line in result.splitlines():
                split_entry = re.split("\(|, ", line)
                if len(split_entry) == 3:

                    label, head_hidx, dep_didx = tuple(split_entry)
                    hh = head_hidx.rfind('-')  # in case word has hyphen
                    #head, hidx = head_hidx[:hh], head_hidx[hh + 1:]
                    head, hidx = head_hidx[:hh], head_hidx[hh + 1:]
                    dh = dep_didx.rfind('-')  # in case word has hyphen
                    #dep, didx = dep_didx[:dh], dep_didx[dh + 1:]
                    dep, didx = dep_didx[:dh], dep_didx[dh + 1:-1]
                    new_dependency = Dep(Head(hidx, head),
                                         label,
                                         Dependent(didx, dep))
                    new_dependencies.append(new_dependency)
        except CalledProcessError as e:
            raise ParserError(e.output)

        return new_dependencies

    def batch_tree2deps(self, trees, **options):

        input_file = tempfile.NamedTemporaryFile()
        str_trees = "\n".join(trees)
        with input_file.file as f:
            f.write(str_trees.encode("utf-8"))

        command = generate_tree2dep_command(corenlp_path=self.corenlp_path, memory=self.memory,
                                            tree_file=input_file.name, **options)

        # creates the xml file of parser output:

        try:
            result = check_output(command, stderr=STDOUT, shell=True)
            output = []
            chunks = result.split("\n\n")
            assert len(chunks) == len(trees) + 1
            for chunk in chunks[:-1]:
                new_dependencies = []
                for line in chunk.split("\n"):
                    split_entry = re.split("\(|, ", line)
                    if len(split_entry) == 3:
                        label, head_hidx, dep_didx = tuple(split_entry)
                        hh = head_hidx.rfind('-')  # in case word has hyphen
                        # head, hidx = head_hidx[:hh], head_hidx[hh + 1:]
                        head, hidx = head_hidx[:hh], head_hidx[hh + 1:]
                        dh = dep_didx.rfind('-')  # in case word has hyphen
                        # dep, didx = dep_didx[:dh], dep_didx[dh + 1:]
                        dep, didx = dep_didx[:dh], dep_didx[dh + 1:-1]
                        new_dependency = Dep(Head(hidx, head),
                                             label,
                                             Dependent(didx, dep))
                        new_dependencies.append(new_dependency)
                output.append(new_dependencies)
        except CalledProcessError as e:
            raise ParserError(e.output)
        return output


# -------------- FUNCTIONS  ---------------------------------------------------------------------------------


def generate_corenlp_command(corenlp_path, memory=None, filelist=None, file=None, outputdir=None, outputformat="",
                             annotators=["tokenize", "ssplit", "pos", "lemma", "ner", " parse", "dcoref"], **kwargs):
    """
    Checks the location of the jar files.
    Spawns the server as a process.
    Input:
        corenlp_path : ""
        memory       : ""
        annotators   : ["tokenize","ssplit","pos","lemma","ner","parse","dcoref"]
        filelist     : None
        file         : None
        outputdir    : None
    """

    jars = ["stanford-corenlp-?.?.?.jar",
            "stanford-corenlp-?.?.?-models.jar",
            "xom.jar",
            "joda-time.jar",
            "jollyday.jar",
            "ejml-?.*.jar",
            "slf4j-api.jar",
            "slf4j-simple.jar",
            "protobuf.jar"]

    java_path = "java"
    classname = "edu.stanford.nlp.pipeline.StanfordCoreNLP"

    # add and check classpaths
    jars = [corenlp_path + "/" + jar for jar in jars]
    missing = [jar for jar in jars if not glob.glob(jar)]
    if missing:
        warnings.warn("Error! Cannot locate: %s" % ', '.join(missing), ImportWarning)

    jars = [glob.glob(jar)[0] for jar in jars]

    # add memory limit on JVM
    if memory:
        limit = "-Xmx%s" % memory
    else:
        limit = ""

    command = "%s %s -cp %s %s" % (java_path, limit, ':'.join(jars), classname)

    annotators = ",".join(annotators)
    command += " -annotators " + annotators

    if file or filelist:

        if file and filelist:
            raise ProcessError("Please give only file or filelist option")

        if file:
            if outputdir:
                command += " -file " + file + " -outputDirectory " + outputdir

            else:
                raise ProcessError("No outputdir given")

        elif filelist:
            if outputdir:
                command += " -filelist " + filelist + " -outputDirectory " + outputdir
            else:
                raise ProcessError("No outputdir given")

    for key, value in kwargs.items():
        command += " -" + key.replace('_', '.') + ' ' + value

    return command


def batch_parser(texts, raw_output=False, corenlp_path=CORENLP_PATH, memory="3g", outputformat="xml",
                annotators = ["tokenize", "ssplit", "pos", "lemma", "ner", "parse", "dcoref"], **kwargs):
    """Gets the job done
    -outputFormat: text, xml, json, conll, conllu
    """

    # First, we change to the directory where we place the xml files from the parser
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    files = []
    file_list = tempfile.NamedTemporaryFile()

    for i, text in enumerate(texts):
        filename = os.path.join(input_dir, "{0}.txt".format(i))
        files.append(filename)
        f = open(filename, "w")
        f.write(text.encode("utf-8"))
        f.close()

    # creating the file list of files to parse
    file_list.write('\n'.join(files))
    file_list.seek(0)

    command = generate_corenlp_command(corenlp_path, memory=memory,
                                       filelist=file_list.name,
                                       outputdir=output_dir,
                                       outputformat=outputformat,
                                       annotators=annotators, **kwargs)

    # print "executing..."
    # print command

    # creates the xml file of parser output:
    call(command, shell=True)

    # reading in the raw xml file:
    try:
        result = []
        for output_file in sorted(os.listdir(output_dir), key=lambda x: int(x.split('.')[0])):
            #print output_file
            with open(output_dir + '/' + output_file, 'r') as xml:
                file_name = re.sub('.xml$', '', os.path.basename(output_file))
                parsed = parse_xml_result(xml.read())
                # add only the first document in the parse
                result.append(parsed[0])
        return result
    finally:
        file_list.close()
        shutil.rmtree(output_dir)


# TO DO
def generate_tree2dep_command(corenlp_path=CORENLP_PATH, memory=None, sent_file=None,
                              tree_file=None, conllx_file=None, **kwargs):
    """
    Usage: java GrammaticalStructure [options]*  [-testGraph]
    options: -basic, -collapsed, -CCprocessed [the default], -collapsedTree, -parseTree, -test, -parserFile file,
             -conllx, -keepPunct, -altprinter -altreader -altreaderfile -originalDependencies
    """

    jars = ["stanford-corenlp-?.?.?.jar",
            "stanford-corenlp-?.?.?-models.jar",
            "xom.jar",
            "joda-time.jar",
            "jollyday.jar",
            "ejml-?.*.jar",
            "slf4j-api.jar",
            "slf4j-simple.jar",
            "protobuf.jar"]

    java_path = "java"
    classname = 'edu.stanford.nlp.trees.EnglishGrammaticalStructure'

    # add and check classpaths
    jars = [corenlp_path + "/" + jar for jar in jars]
    missing = [jar for jar in jars if not glob.glob(jar)]
    if missing:
        warnings.warn("Error! Cannot locate: %s" % ', '.join(missing), ImportWarning)

    jars = [glob.glob(jar)[0] for jar in jars]

    # add memory limit on JVM
    if memory:
        limit = "-Xmx%s" % memory
    else:
        limit = ""

    command = "%s %s -cp %s %s" % (java_path, limit, ':'.join(jars), classname)

    if not any([sent_file, tree_file, conllx_file]):
        raise ProcessError("Please give only file or filelist option")
    if sent_file:
        command += " -sentFile " + sent_file
    if tree_file:
        command += " -treeFile " + tree_file
    if conllx_file:
        command += " -conllxFile " + conllx_file

    for key, value in kwargs.items():
        command += " -" + key.replace('_', '.') + ' ' + value

    return command


def parse_interactive_shell_result(text):
    """ This is the nasty bit of code to interact with the command-line
    interface of the CoreNLP tools.  Takes a string of the parser results
    and then returns a Python list of dictionaries, one for each parsed
    sentence.
    """

    def parse_bracketed(i, s):
        '''Parse word features [abc=... def = ...]
        Also manages to parse out features that have XML within them
        '''
        word = None
        attrs = {}
        temp = {}
        # Substitute XML tags, to replace them later
        for i, tag in enumerate(re.findall(r"(<[^<>]+>.*<\/[^<>]+>)", s)):
            temp["^^^%d^^^" % i] = tag
            s = s.replace(tag, "^^^%d^^^" % i)
        # Load key-value pairs, substituting as necessary
        for attr, val in re.findall(r"([^=\s]*)=([^=\s]*)", s):
            if val in temp:
                val = temp[val]
            if attr == 'Text':
                word = val
            else:
                attrs[attr] = val
        return Token(i,
                     word,
                     attrs.get('PartOfSpeech'),
                     attrs.get('Lemma'),
                     attrs.get('NamedEntityTag'),
                     attrs.get('CharacterOffsetBegin'),
                     attrs.get('CharacterOffsetEnd'))

    state = STATE_START
    new_sentences = []

    if sys.version_info[0] < 3 and isinstance(text, str) or \
       sys.version_info[0] >= 3 and isinstance(text, bytes):
        text = text.decode('utf-8')

    for line in text.split('\n'):
        line = line.strip()

        if line.startswith("Sentence #"):
            if state == STATE_START:
                    new_tokens = []
                    syntax_tree = []
                    new_dependencies = []
            else:
                s = Sentence(new_tokens, syntax_tree, new_dependencies)
                new_sentences.append(s)
                new_tokens = []
                syntax_tree = []
                new_dependencies = []
            state = STATE_TEXT

        elif state == STATE_TEXT:
            state = STATE_WORDS

        elif state == STATE_WORDS:
            if not line.startswith("[Text="):
                state = STATE_TREE
                syntax_tree.append(line)
                #raise ParserError('Parse error. Could not find "[Text=" in: %s' % line)
            for i, s in enumerate(WORD_PATTERN.findall(line)):
                new_tokens.append(parse_bracketed(i, s))

        elif state == STATE_TREE:
            if len(line) == 0:
                state = STATE_DEPENDENCY
                syntax_tree = " ".join(syntax_tree)
            else:
                #print syntax_tree
                syntax_tree.append(line)

        elif state == STATE_DEPENDENCY:
            if len(line) == 0:
                state = STATE_COREFERENCE
            else:
                split_entry = re.split("\(|, ", line[:-1])
                if len(split_entry) == 3:
                    label, head_hidx, dep_didx = tuple(split_entry)
                    hh = head_hidx.rfind('-') # in case word has hyphen
                    head, hidx = head_hidx[:hh], head_hidx[hh+1:]
                    dh = dep_didx.rfind('-') # in case word has hyphen
                    dep, didx = dep_didx[:dh], dep_didx[dh+1:]
                    new_dependency = Dep(Head(hidx, head),
                                         label,
                                         Dependent(didx, dep))
                    new_dependencies.append(new_dependency)

        elif state == STATE_COREFERENCE:
            pass

    s = Sentence(new_tokens, syntax_tree, new_dependencies)
    new_sentences.append(s)
    return new_sentences


def parse_xml_result(xml):
    # Turning the raw xml into a raw python dictionary:

    result = et.fromstring(xml)
    new_documents = []
    for document in result.findall('document'):
        new_sentences = []
        sentences = document[0]
        for sentence in sentences.findall('sentence'):

            # extract syntax https://www.google.cl/?gws_rd=ssltree
            syntax_tree = sentence.find('parse')
            if syntax_tree is not None:
                syntax_tree = syntax_tree.text

            # extract dependencies
            new_dependencies = []
            dependencies = sentence.find('dependencies')
            if dependencies is not None:
                for dep in dependencies:

                    label = dep.get('type')

                    head = dep.find('governor')
                    index = head.get('idx')
                    word = head.text
                    new_head = Head(index, word)

                    dependent = dep.find('dependent')
                    index = dependent.get('idx')
                    word = dependent.text
                    new_dependent = Dependent(index, word)

                    new_dependencies.append(Dep(new_head, label, new_dependent))

            # extract Tokens
            tokens = sentence[0]
            new_tokens = []
            for token in tokens.findall('token'):
                index = token.get('id')
                word = token.find('word').text

                # get start offset
                start = token.find('CharacterOffsetBegin')
                if start is not None:
                    start = int(start.text)

                # get end offset
                end = token.find('CharacterOffsetEnd')
                if end is not None:
                    end = int(end.text)

                # get POS tag
                pos = token.find('POS')
                if pos is not None:
                    pos = pos.text

                # get lemma
                lemma = token.find('lemma')
                if lemma is not None:
                    lemma = lemma.text

        # print "executing..."
        # print command
                # get NER tag
                ner = token.find('NER')
                if ner is not None:
                    ner = ner.text

                new_tokens.append(Token(index, word, pos, lemma, ner,
                                        start, end))

            new_sentence = Sentence(new_tokens, syntax_tree, new_dependencies)
            new_sentences.append(new_sentence)
        new_documents.append(new_sentences)

    return new_documents


def main():
    parser = CoreNLP(annotators = ['tokenize','ssplit','pos'],
                     tokenize_model="PTBTokenizer",
                     ssplit_isOneSentence="true")

    result = parser.parse("Hello, my name is Edison. I come from Chile")

    print result

    result = batch_parser(["Hello, my name is Edison"],
                          annotators=['tokenize','ssplit','pos','lemma','parse'])

    for document in result:
        for sentence in document:
            print sentence.tokens
            print sentence.syntax_tree
            print sentence.dependencies

if __name__ == '__main__':
    main()
