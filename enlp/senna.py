#!/usr/bin/python
# -*- coding: utf-8 -*-

import subprocess
import os
from platform import architecture, system
from collections import namedtuple

from settings import SENNA_PATH

# -------------- RESULTS ------------------------------------------------------

Sentence = namedtuple('Sentence', ['tokens',
                                   'syntax_tree',
                                   'semantic_role_labels'])
Token = namedtuple("Token", ['index', 'word', 'pos', 'chunk', 'ner',
                             'start', 'end'])

# -------------- MAIN CLASS ---------------------------------------------------


class Senna:
    """
    A general interface of the SENNA/Stanford Dependency Extractor pipeline 
    that supports any of the operations specified in SUPPORTED_OPERATIONS.

    SUPPORTED_OPERATIONS: It provides Part of Speech Tags, Semantic Role Labels,
    Shallow Parsing (Chunking), Named Entity
    Recognisation (NER), Dependency Parse and Syntactic Constituency Parse.

    Applying multiple operations at once has the speed advantage. For example,
    senna v3.0 will calculate the POS tags in case you are extracting the named
    entities. Applying both of the operations will cost only the time of
    extracting the named entities. Same is true for dependency Parsing.

    SENNA pipeline has a fixed maximum size of the sentences that it can read.
    By default it is 1024 token/sentence. If you have larger sentences, changing
    the MAX_SENTENCE_SIZE value in SENNA_main.c should be considered and your
    system specific binary should be rebuilt. Otherwise this could introduce
    misalignment errors.
    """

    def __init__(self, senna_path=SENNA_PATH):
        self.senna_dir = senna_path

    def parse(self, sentence, args=["pos", "chk", "ner", "srl", "psg"]):
        """
        SENNA supports the following options:

            -h             : Display an inline help.
            -verbose       : Display model informations (on the standard error output, so it 
                             does not mess up the tag outputs).
            -notokentags   : Do not output tokens (first output column).
            -offsettags    : Output start/end character offset (in the sentence), for each token.
            -iobtags       : Output IOB tags instead of IOBES.
            -brackettags   : Output 'bracket' tags instead of IOBES.
            -path <path>   : Specify the path to the SENNA data/ and hash/ directories, if you do not run
                             SENNA in its original directory. The path must end by "/".
            -usrtokens     : Use user's tokens (space separated) instead of SENNA tokenizer.
            -posvbs        : Use verbs outputed by the POS tagger instead of SRL style verbs for SRL task.
                             You might want to use this, as the SRL training task ignore some verbs 
                             (many "be" and "have") which might be not what you want.
            -usrvbs <file> : Use user's verbs (given in <file>) instead of SENNA verbs for SRL task. 
                             The file must contain one line per token, with an empty line 
                             between each sentence. A line which is not a "-" corresponds to a verb.
            -pos/-chk/-ner/-srl/-psg
                           : Instead of outputing tags for all tasks, SENNA will output tags
                             for the specified (one or more) tasks.

        """
        senna_tags = self.get_tags(sentence, args)
        return self.get_annotations(senna_tags, args)

    def batch_parse(self, sentences, args=["pos", "chk", "ner", "srl", "psg"]):
        """
        SENNA supports the following options:

            -h             : Display an inline help.
            -verbose       : Display model informations (on the standard error output, so it 
                             does not mess up the tag outputs).
            -notokentags   : Do not output tokens (first output column).
            -offsettags    : Output start/end character offset (in the sentence), for each token.
            -iobtags       : Output IOB tags instead of IOBES.
            -brackettags   : Output 'bracket' tags instead of IOBES.
            -path <path>   : Specify the path to the SENNA data/ and hash/ directories, if you do not run
                             SENNA in its original directory. The path must end by "/".
            -usrtokens     : Use user's tokens (space separated) instead of SENNA tokenizer.
            -posvbs        : Use verbs outputed by the POS tagger instead of SRL style verbs for SRL task.
                             You might want to use this, as the SRL training task ignore some verbs 
                             (many "be" and "have") which might be not what you want.
            -usrvbs <file> : Use user's verbs (given in <file>) instead of SENNA verbs for SRL task. 
                             The file must contain one line per token, with an empty line 
                             between each sentence. A line which is not a "-" corresponds to a verb.
            -pos/-chk/-ner/-srl/-psg
                           : Instead of outputing tags for all tasks, SENNA will output tags
                             for the specified (one or more) tasks.

        """
        senna_tags_batch = self.get_tags_batch(sentences, args)
        return self.get_batch_annotations(senna_tags_batch, args)

    def get_tags(self, sentence, args):
        # senna interprets \n as new sentence so we replace it
        input_data = sentence.replace('\n', ' ').replace('\r', '').encode('utf8')
        package_directory = self.senna_dir
        os_name = system()
        executable = ""
        if os_name == 'Linux':
            bits = architecture()[0]
            if bits == '64bit':
                executable = 'senna-linux64'
            elif bits == '32bit':
                executable = 'senna-linux32'
        else:
            executable = 'senna'
        if os_name == 'Windows':
            executable = 'senna-win32.exe'
        if os_name == 'Darwin':
            executable = 'senna-osx'
        senna_executable = os.path.join(package_directory, executable)
        pargs = [senna_executable]

        if "psg" in args and "pos" not in args:
            args.append("pos")

        for item in args:
            pargs.append("-"+item)

        cwd = os.getcwd()
        os.chdir(package_directory)
        p = subprocess.Popen(pargs, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        senna_stdout = p.communicate(input=input_data)[0]
        os.chdir(cwd)
        return senna_stdout

    def get_tags_batch(self, sentences, args):
        input_data = ""
        for sentence in sentences:
            # senna interprets \n as new sentence so we replace it
            sentence = sentence.replace('\n', ' ').replace('\r', '')
            input_data += sentence.encode('utf8')+"\n"
        input_data = input_data[:-1]
        package_directory = self.senna_dir
        os_name = system()
        executable = ""
        if os_name == 'Linux':
            bits = architecture()[0]
            if bits == '64bit':
                executable = 'senna-linux64'
            elif bits == '32bit':
                executable = 'senna-linux32'
        else:
            executable = 'senna'
        if os_name == 'Windows':
            executable = 'senna-win32.exe'
        if os_name == 'Darwin':
            executable = 'senna-osx'
        senna_executable = os.path.join(package_directory, executable)
        pargs = [senna_executable]

        if "psg" in args and "pos" not in args:
            args.append("pos")

        for item in args:
            pargs.append("-"+item)

        cwd = os.getcwd()
        os.chdir(package_directory)
        p = subprocess.Popen(pargs, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        senna_stdout = p.communicate(input=input_data)[0]
        os.chdir(cwd)
        return senna_stdout.split("\n\n")[0:-1]

    def get_batch_annotations(self, senna_tags_batch, args):
        annotations = []
        for senna_tags in senna_tags_batch:
            annotations += [self.get_annotations(senna_tags, args, batch=True)]
        return annotations

    def get_annotations(self, senna_tags, args, batch=False):
        senna_tags = map(lambda x: x.strip(), senna_tags.split("\n"))
        no_verbs = len(senna_tags[0].split("\t"))-(len(args)+1)

        words = []
        locs = []
        pos = []
        chunk = []
        ner = []
        verb = []
        srls = []
        syn = []

        # get easy tags
        limit = len(senna_tags)+1 if batch else -2
        for senna_tag in senna_tags[0:limit]:
            senna_tag = senna_tag.split("\t")

            words += [senna_tag[0].strip()]
            index = 1

            if "offsettags" in args:
                start, end = senna_tag[index].split()
                locs.append((int(start.strip()),
                             int(end.strip())))
                index += 1

            if 'pos' in args:
                pos += [senna_tag[index].strip()]
                index += 1
            else:
                pos.append(None)

            if 'chk' in args:
                chunk += [senna_tag[index].strip()]
                index += 1
            else:
                chunk.append(None)

            if 'ner' in args:
                ner += [senna_tag[index].strip()]
                index += 1
            else:
                ner.append(None)

            if "srl" in args:
                verb += [senna_tag[index].strip()]

                add = 0 if 'psg' in args else 1
                srl = []
                start = len(args) + add
                end = len(args)+no_verbs+add

                for i in range(start, end):
                    srl += [senna_tag[i].strip()]

                srls += [tuple(srl)]

            if 'psg' in args:
                syn += [senna_tag[-1]]

        # get syntax tree
        syntax_tree = ""
        if 'psg' in args:
            for (w, s, p) in zip(words, syn, pos):
                syntax_tree += s.replace("*", "("+p+" "+w+")")

        # get roles
        roles = []
        if "srl" in args:
            for j in range(no_verbs):
                role = {}
                i = 0
                temp = ""
                curr_labels = map(lambda x: x[j], srls)
                for curr_label in curr_labels:
                    splits = curr_label.split("-")
                    if(splits[0] == "S"):
                        if(len(splits) == 2):
                            if(splits[1] == "V"):
                                role[splits[1]] = words[i]
                            else:
                                if splits[1] in role:
                                    role[splits[1]] += " " + words[i]
                                else:
                                    role[splits[1]] = words[i]
                    elif(len(splits) == 3):
                        if splits[1]+"-"+splits[2] in role:
                            role[splits[1]+"-"+splits[2]] += " " + words[i]
                        else:
                            role[splits[1] + "-" + splits[2]] = words[i]
                    elif(splits[0] == "B"):
                        temp = temp + " " + words[i]
                    elif(splits[0] == "I"):
                            temp = temp + " " + words[i]
                    elif(splits[0] == "E"):
                        temp = temp + " " + words[i]
                        if(len(splits) == 2):
                            if(splits[1] == "V"):
                                role[splits[1]] = temp.strip()
                            else:
                                if splits[1] in role:
                                    role[splits[1]] += " " +temp
                                    role[splits[1]] = role[splits[1]].strip()
                                else:
                                    role[splits[1]] = temp.strip()
                        elif(len(splits) == 3):
                            if splits[1]+"-"+splits[2] in role:
                                role[splits[1] + "-" + splits[2]] += " " + temp
                                role[splits[1] + "-" + splits[2]] = role[splits[1] + "-" + splits[2]].strip()
                            else:
                                role[splits[1] + "-" + splits[2]] = temp.strip()
                                temp = ""
                    i += 1
                if("V" in role):
                    roles += [role]

        tokens = []
        for i, word in enumerate(words):
            if locs:
                tokens.append(Token(i, word, pos[i],
                                    chunk[i], ner[i],
                                    locs[i][0], locs[i][1]))
            else:
                tokens.append(Token(i, word, pos[i],
                                    chunk[i], ner[i],
                                    None, None))

        # verbs = filter(lambda x: x!="-",verb)

        return Sentence(tokens, syntax_tree, roles)


def main():
    senna = Senna()
    result = senna.batch_parse(["Hello my name is John.", "I hate football."])
    for sentence in result:
        print sentence.tokens
        print sentence.syntax_tree
        print sentence.semantic_role_labels


if __name__ == '__main__':
    main()
