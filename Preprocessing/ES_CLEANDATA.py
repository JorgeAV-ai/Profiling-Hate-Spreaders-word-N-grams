import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import xml.etree.ElementTree as et 
from bs4 import BeautifulSoup

# !pip install wordcloud
from wordcloud import WordCloud

from nltk.corpus import stopwords
import string


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import  TfidfVectorizer, CountVectorizer, HashingVectorizer, TfidfTransformer


from  nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('spanish')
import nltk

# Cleaning Data

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    return re.sub(emoji_pattern,' #emoji# ',text)

spanish_stopword = stopwords.words('spanish')
spanish_stopword.remove("no")
spanish_stopword.extend(["url","user","hashtag","emoji","rt"])
# print(spanish_stopword)

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

def remove_stopwords(text):
    text = [word for word in text.split() if word not in spanish_stopword]
    return text

def Clean_tweet(text):
#   lowerize
    Cleantweet = text.lower()
#     remove emojis
    Cleantweet = remove_emoji(Cleantweet)
#     remove punctuaction '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    Cleantweet = remove_punct(Cleantweet)
#     delete stopwords
    Cleantweet = remove_stopwords(Cleantweet)
    Cleantweet = " ".join([word for word in Cleantweet])
    return Cleantweet

def dataset():
    path = "pan21/es/"
    rows = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            if 'xml' in filename:
                tweets_raw = []
                tree = et.parse(path+filename)
                root = tree.getroot()
                classe = root.attrib['class']
                readXML = BeautifulSoup(open(path+filename),'xml')
                tweets = readXML.findAll('document')
                for tweet in tweets:
                    tweets_raw.append(tweet.text)
                tweets_raw = ' '.join(tweets_raw)
                df_cols = ["ID","class","tweets"]
                rows.append({"ID": filename.split(".")[0],"class": classe, "tweets":tweets_raw})

    df = pd.DataFrame(rows, columns = df_cols)
    df.to_csv('archivos/ES_Dataset.csv', sep='\t', index=False)

def Clean_dataset():
    path = "test/es/"
    rows = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            if 'xml' in filename:
                tweets_raw = []
                tree = et.parse(path+filename)
                root = tree.getroot()
                classe = root.attrib['class']
                readXML = BeautifulSoup(open(path+filename),'xml')
                tweets = readXML.findAll('document')
                for tweet in tweets:
                    tweet = Clean_tweet(tweet.text)
                    tweets_raw.append(tweet)
                tweets_raw = ' '.join(tweets_raw)
                df_cols = ["ID","class","tweets"]
                rows.append({"ID": filename.split(".")[0],"class": classe, "tweets":tweets_raw})

    df = pd.DataFrame(rows, columns = df_cols)
    df.to_csv('archivos/ES_CleanDataset_test.csv', sep='\t', index=False)


if __name__ == '__main__':
    #  v0 dataset convert to Dataframe
    # dataset()
    # v1 clear dataset punctuaction, stopwords(without "no", including url, user, hashtag, emoji, rt)
    Clean_dataset()
    