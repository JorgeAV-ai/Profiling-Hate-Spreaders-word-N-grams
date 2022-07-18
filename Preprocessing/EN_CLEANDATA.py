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
stemmer = SnowballStemmer('english')
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

# Contractions:
def contrations(text):
    tweet = re.sub(r"he's", "he is", tweet)
    tweet = re.sub(r"there's", "there is", tweet)
    tweet = re.sub(r"We're", "We are", tweet)
    tweet = re.sub(r"That's", "That is", tweet)
    tweet = re.sub(r"won't", "will not", tweet)
    tweet = re.sub(r"they're", "they are", tweet)
    tweet = re.sub(r"Can't", "Cannot", tweet)
    tweet = re.sub(r"wasn't", "was not", tweet)
    tweet = re.sub(r"don\x89Ûªt", "do not", tweet)
    tweet = re.sub(r"aren't", "are not", tweet)
    tweet = re.sub(r"isn't", "is not", tweet)
    tweet = re.sub(r"What's", "What is", tweet)
    tweet = re.sub(r"haven't", "have not", tweet)
    tweet = re.sub(r"hasn't", "has not", tweet)
    tweet = re.sub(r"There's", "There is", tweet)
    tweet = re.sub(r"He's", "He is", tweet)
    tweet = re.sub(r"It's", "It is", tweet)
    tweet = re.sub(r"You're", "You are", tweet)
    tweet = re.sub(r"I'M", "I am", tweet)
    tweet = re.sub(r"shouldn't", "should not", tweet)
    tweet = re.sub(r"wouldn't", "would not", tweet)
    tweet = re.sub(r"i'm", "I am", tweet)
    tweet = re.sub(r"I\x89Ûªm", "I am", tweet)
    tweet = re.sub(r"I'm", "I am", tweet)
    tweet = re.sub(r"Isn't", "is not", tweet)
    tweet = re.sub(r"Here's", "Here is", tweet)
    tweet = re.sub(r"you've", "you have", tweet)
    tweet = re.sub(r"you\x89Ûªve", "you have", tweet)
    tweet = re.sub(r"we're", "we are", tweet)
    tweet = re.sub(r"what's", "what is", tweet)
    tweet = re.sub(r"couldn't", "could not", tweet)
    tweet = re.sub(r"we've", "we have", tweet)
    tweet = re.sub(r"it\x89Ûªs", "it is", tweet)
    tweet = re.sub(r"doesn\x89Ûªt", "does not", tweet)
    tweet = re.sub(r"It\x89Ûªs", "It is", tweet)
    tweet = re.sub(r"Here\x89Ûªs", "Here is", tweet)
    tweet = re.sub(r"who's", "who is", tweet)
    tweet = re.sub(r"I\x89Ûªve", "I have", tweet)
    tweet = re.sub(r"y'all", "you all", tweet)
    tweet = re.sub(r"can\x89Ûªt", "cannot", tweet)
    tweet = re.sub(r"would've", "would have", tweet)
    tweet = re.sub(r"it'll", "it will", tweet)
    tweet = re.sub(r"we'll", "we will", tweet)
    tweet = re.sub(r"wouldn\x89Ûªt", "would not", tweet)
    tweet = re.sub(r"We've", "We have", tweet)
    tweet = re.sub(r"he'll", "he will", tweet)
    tweet = re.sub(r"Y'all", "You all", tweet)
    tweet = re.sub(r"Weren't", "Were not", tweet)
    tweet = re.sub(r"Didn't", "Did not", tweet)
    tweet = re.sub(r"they'll", "they will", tweet)
    tweet = re.sub(r"they'd", "they would", tweet)
    tweet = re.sub(r"DON'T", "DO NOT", tweet)
    tweet = re.sub(r"That\x89Ûªs", "That is", tweet)
    tweet = re.sub(r"they've", "they have", tweet)
    tweet = re.sub(r"i'd", "I would", tweet)
    tweet = re.sub(r"should've", "should have", tweet)
    tweet = re.sub(r"You\x89Ûªre", "You are", tweet)
    tweet = re.sub(r"where's", "where is", tweet)
    tweet = re.sub(r"Don\x89Ûªt", "Do not", tweet)
    tweet = re.sub(r"we'd", "we would", tweet)
    tweet = re.sub(r"i'll", "I will", tweet)
    tweet = re.sub(r"weren't", "were not", tweet)
    tweet = re.sub(r"They're", "They are", tweet)
    tweet = re.sub(r"Can\x89Ûªt", "Cannot", tweet)
    tweet = re.sub(r"you\x89Ûªll", "you will", tweet)
    tweet = re.sub(r"I\x89Ûªd", "I would", tweet)
    tweet = re.sub(r"let's", "let us", tweet)
    tweet = re.sub(r"it's", "it is", tweet)
    tweet = re.sub(r"can't", "cannot", tweet)
    tweet = re.sub(r"don't", "do not", tweet)
    tweet = re.sub(r"you're", "you are", tweet)
    tweet = re.sub(r"i've", "I have", tweet)
    tweet = re.sub(r"that's", "that is", tweet)
    tweet = re.sub(r"i'll", "I will", tweet)
    tweet = re.sub(r"doesn't", "does not", tweet)
    tweet = re.sub(r"i'd", "I would", tweet)
    tweet = re.sub(r"didn't", "did not", tweet)
    tweet = re.sub(r"ain't", "am not", tweet)
    tweet = re.sub(r"you'll", "you will", tweet)
    tweet = re.sub(r"I've", "I have", tweet)
    tweet = re.sub(r"Don't", "do not", tweet)
    tweet = re.sub(r"I'll", "I will", tweet)
    tweet = re.sub(r"I'd", "I would", tweet)
    tweet = re.sub(r"Let's", "Let us", tweet)
    tweet = re.sub(r"you'd", "You would", tweet)
    tweet = re.sub(r"It's", "It is", tweet)
    tweet = re.sub(r"Ain't", "am not", tweet)
    tweet = re.sub(r"Haven't", "Have not", tweet)
    tweet = re.sub(r"Could've", "Could have", tweet)
    tweet = re.sub(r"youve", "you have", tweet)  
    tweet = re.sub(r"donå«t", "do not", tweet)   
    return tweet     


def Clean_tweet(text):
#   lowerize
    Cleantweet = text.lower()
#     remove emojis
    Cleantweet = remove_emoji(Cleantweet)
#     remove punctuaction '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    Cleantweet = remove_punct(Cleantweet)
#     delete stopwords
    Cleantweet = remove_stopwords(Cleantweet)
    Cleantweet = contrations(Cleantweet)
    Cleantweet = " ".join([word for word in Cleantweet])
    return Cleantweet

def dataset():
    path = "test/en/"
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
    df.to_csv('archivos/EN_Dataset.csv', sep='\t', index=False)

def Clean_dataset():
    path = "test/en/"
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
    df.to_csv('archivos/EN_CleanDataset_test.csv', sep='\t', index=False)



if __name__ == '__main__':
    #  v0 dataset convert to Dataframe
    # dataset()
    # v1 clear dataset punctuaction, stopwords(without "no", including url, user, hashtag, emoji, rt)
    Clean_dataset()
    
