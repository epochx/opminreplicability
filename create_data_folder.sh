#!/usr/bin/env bash

DATAPATH=$1

mkdir -p "$DATAPATH"/data/{pickle,corpora}
mkdir -p "$DATAPATH"/data/lexicon/liu
mkdir -p "$DATAPATH"/data/corpora/stopwords
mkdir -p "$DATAPATH"/data/corpora/opinion/{liu,semeval-absa-2014,youtube}

# Liu Customer Review Dataset
wget http://www.cs.uic.edu/~liub/FBS/CustomerReviewData.zip
unzip ./CustomerReviewData.zip
mv ./customer\ review\ data/Apex\ AD2600\ Progressive-scan\ DVD\ player.txt "$DATAPATH"/data/corpora/opinion/liu/apex_dvd_player.txt
mv ./customer\ review\ data/Nikon\ coolpix\ 4300.txt "$DATAPATH"/data/corpora/opinion/liu/nikon_camera.txt
mv ./customer\ review\ data/Canon\ G3.txt "$DATAPATH"/data/corpora/opinion/liu/canon_camera.txt
mv ./customer\ review\ data/Nokia\ 6610.txt "$DATAPATH"/data/corpora/opinion/liu/nokia_cellphone.txt
mv ./customer\ review\ data/Creative\ Labs\ Nomad\ Jukebox\ Zen\ Xtra\ 40GB.txt "$DATAPATH"/data/corpora/opinion/liu/creative_mp3_player.txt

# stanford stopwords
wget https://raw.githubusercontent.com/stanfordnlp/CoreNLP/master/data/edu/stanford/nlp/patterns/surface/stopwords.txt
mv ./stopwords.txt "$DATAPATH"/data/stopwords/stanford_stopwords.txt

# nltk stopwords
wget https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip
unzip -p stopwords.zip stopwords/english >./data/corpora/stopwords/nltk_stopwords.txt

# opinion words
wget http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar
unrar e ./opinion-lexicon-English.rar
mv ./positive-words.txt "$DATAPATH"/data/lexicon/liu/positive_words.txt
mv ./negative-words.txt "$DATAPATH"/data/lexicon/liu/negative_words.txt

rm -r CustomerReviewData.zip opinion-lexicon-English.rar customer\ review\ data
rm stopwords.zip

echo "Patching corpora"
cd "$DATAPATH"/data/corpora/opinion/liu

sed -i '155s/feature/feature[+2]/' nokia_cellphone.txt
sed -i '480s/look{+1]/look[+1]/' nokia_cellphone.txt
sed -i '79s/^feature\[+2\]\,\s/feature[+2]/' canon_camera.txt
sed -i '157s/look##/look[+2]##/' creative_mp3_player.txt
sed -i '334s/setup\[2\]/setup[+2]/' creative_mp3_player.txt
sed -i '485s/#/##/' apex_dvd_player.txt
