import pandas as pd
from numpy import mean
from numpy import std
import numpy as np
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics import accuracy_score


# Testing results with Hate lexicon 
def hate_lexicon(df):
    f = open("lexiconES.txt","r")
    words = []
    res_per_tweet = []
    for line in f.readlines():
        if not line.startswith("#"):
            try:
                w = line.split()[0]
                words.append(w)
            except:
                pass

    for tweet in df:
        pos = 0
        for word in tweet.split():
            if word in words:
                    pos+=1
        res_per_tweet.append([pos])
    return np.matrix(res_per_tweet)


# Voting classifier 
def models_voting():
    estimators = [
        ('lr',make_pipeline(TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True,analyzer='word',min_df=9,ngram_range=(1,2)),LogisticRegression(C=1000,penalty='l2', solver='liblinear',verbose=0, random_state=0))),
        ('nb',make_pipeline(TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True,analyzer='word',min_df=5,ngram_range=(1,2)),MultinomialNB(alpha=0.25,fit_prior=True))),
        ('svm',make_pipeline(TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True,analyzer= "word",ngram_range=(1,6), min_df = 0.05), SVC(C=100, kernel = "linear", gamma = "scale",probability=True)))
    ]
    ensemble = VotingClassifier(estimators=estimators,voting='hard')
    return ensemble

def get_models():
    models= {}
    models['LR'] = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True,analyzer='word',min_df=9,ngram_range=(1,2))),('lr', LogisticRegression(C=1000,penalty='l2', solver='liblinear',verbose=0, random_state=0))])
    models['NB'] = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True,analyzer='word',min_df=5,ngram_range=(1,2))),('nb', MultinomialNB(alpha=0.25,fit_prior=True))])
    models['KNN'] = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True,analyzer='word',min_df=5,ngram_range=(2,2))),('knn', KNeighborsClassifier(n_jobs=-1,leaf_size=20,metric='euclidean',n_neighbors=10,p=1,weights='distance'))])
    models['RF'] = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True,analyzer='word',min_df=0.05,ngram_range=(1,6))),('rf', RandomForestClassifier(criterion='gini',max_depth=3,min_samples_split=6,random_state=0))])
    models['SVC'] = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True,analyzer= "word",ngram_range=(1,6), min_df = 0.05)), ('svm', SVC(C=100, kernel = "linear", gamma = "scale",probability=True))])
    return models

def evaluate_model(model,X,ourcv):
    train_vectors = model[0].fit_transform(X["tweets"])
    train_palabras_con_polaridad = hate_lexicon(X["tweets"])
    train_vectors = sparse.hstack((train_vectors,train_palabras_con_polaridad))
    res = cross_val_score(model[1],train_vectors,X["class"],cv=ourcv)

    print(res)
    return mean(res)


if __name__ == '__main__':
    print("Starting...")
    dfClean = pd.read_csv("archivos/ES_CleanDataset.csv",sep='\t')
    dfCleantest = pd.read_csv("archivos/ES_CleanDataset_test.csv",sep='\t')
    models = get_models()
    for name, model in models.items():
        print(model)
        train_vectors = model[0].fit_transform(dfClean["tweets"])
        train_palabras_con_polaridad = hate_lexicon(dfClean["tweets"])
        train_vectors = sparse.hstack((train_vectors,train_palabras_con_polaridad))
        model[1].fit(train_vectors,dfClean["class"])
        test_vectors = model[0].transform(dfCleantest["tweets"])
        test_palabras_con_polaridad = hate_lexicon(dfCleantest["tweets"])
        test_vectors = sparse.hstack((test_vectors,test_palabras_con_polaridad))
        res = model[1].predict(test_vectors)
        print(classification_report(dfCleantest["class"],res))
        print(name)

    model_ensemble = models_voting()
    model_fitted = model_ensemble.fit(dfClean["tweets"],dfClean["class"])
    res = model_fitted.predict(dfCleantest["tweets"])
    print(classification_report(dfCleantest["class"],res))