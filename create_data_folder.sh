#!/usr/bin/env bash

DATAPATH=$1

mkdir -p "$DATAPATH"/data/{pickle,corpora}
mkdir -p "$DATAPATH"/data/lexicon/liu
mkdir -p "$DATAPATH"/data/stopwords
mkdir -p "$DATAPATH"/data/corpora/opinion/{semeval-absa-2014,youtube}

# Liu Customer Review Dataset
wget http://www.cs.uic.edu/~liub/FBS/CustomerReviewData.zip
unzip ./CustomerReviewData.zip
mv ./customer_review_data/Apex AD2600 Progressive-scan DVD player.txt "$DATAPATH"/data/corpora/opinion/apex_dvd_player.txt
mv ./customer_review_data/Nikon coolpix 4300.txt "$DATAPATH"/data/corpora/opinion/nikon_camera.txt
mv ./customer_review_data/Canon G3.txt "$DATAPATH"/data/corpora/opinion/canon_camera.txt
mv ./customer_review_data/Nokia 6610.txt "$DATAPATH"/data/corpora/opinion/nokia_cellphone.txt
mv ./customer_review_data/Creative Labs Nomad Jukebox Zen Xtra 40GB.txt "$DATAPATH"/data/corpora/opinion/creative_mp3_player.txt

# stanford stopwords
wget https://raw.githubusercontent.com/stanfordnlp/CoreNLP/master/data/edu/stanford/nlp/patterns/surface/stopwords.txt
mv ./stopwords.txt "$DATAPATH"/data/stopwords/stanford_stopwords.txt

# nltk stopwords
cp ~/nltk_data/stopwords.txt "$DATAPATH"/data/stopwords/nltk_stopwords.txt

# opinion words
wget http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar
unzip ./opinion-lexicon-English.rar
mv ./opinion-lexicon-English/positive_words.txt "$DATAPATH"/data/lexicon/liu/positive_words.txt
mv ./opinion-lexicon-English/negative_words.txt "$DATAPATH"/data/lexicon/liu/negative_words.txt
