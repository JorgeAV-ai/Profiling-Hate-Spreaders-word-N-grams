import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import  TfidfVectorizer, CountVectorizer, HashingVectorizer
import pickle

import warnings
warnings.filterwarnings("ignore")

def nbtfidf(df):
    pipeline = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True)),
                    ('nb', MultinomialNB())])
    
    parameters={"vect__analyzer": ["word"],
                "vect__ngram_range": [(1,1),(1,2),(2,2)],
                "vect__min_df":[4,5,6,7,8,9,10],
                "nb__alpha":[0.1,0.25,0.5,0.75,1.0],
                "nb__fit_prior":[True,False]
                }

    grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(10,shuffle=True, random_state=0),n_jobs=-1,verbose=2)

    lr_tfidf =grid_search.fit(df["tweets"],df['class'])
    dflr=pd.DataFrame()
    dflr['params']=lr_tfidf.cv_results_['params']
    dflr['scores']=lr_tfidf.cv_results_['mean_test_score']
    dflr.to_csv('archivos/ES_NB_tfidf_Dataset_results.tsv', sep='\t', index=False)

    # given the best hyperparameters
    #y_true, y_pred = X_test["class"], lr_tfidf.predict(X_test["tweets"])
    #print(classification_report(y_true, y_pred))


def nbtfidf2(df):
    pipeline = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True)),
                    ('nb', MultinomialNB())])
    
    parameters={"vect__analyzer": ["word","char","char_wb"],
                "vect__ngram_range": [(1,1),(1,2),(2,2),(2,3),(3,3),(3,4),(4,4)],
                "vect__min_df":[1,2,3,4,5,6,7,8,9,10],
                "nb__alpha":[0.0,0.25,0.5,0.75,1.0],
                "nb__fit_prior":[True,False]
                }

    grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(5,shuffle=True, random_state=0),n_jobs=-1)

    lr_tfidf =grid_search.fit(df["tweets"],df['class'])
    dflr=pd.DataFrame()
    dflr['params']=lr_tfidf.cv_results_['params']
    dflr['scores']=lr_tfidf.cv_results_['mean_test_score']
    dflr.to_csv('archivos/ES_NB_tfidf_CleanDataset_results.tsv', sep='\t', index=False)

    # given the best hyperparameters
    y_true, y_pred = X_test["class"], lr_tfidf.predict(X_test["tweets"])
    print(classification_report(y_true, y_pred))

# def nbcount(df):
#     pipeline = Pipeline([('vect', CountVectorizer()),
#                     ('nb', MultinomialNB())])
    
#     parameters={"vect__analyzer": ["word","char","char_wb"],
#                 "vect__ngram_range": [(1,1),(1,2),(2,2),(2,3),(3,3),(3,4),(4,4)],
#                 "vect__min_df":[1,2,3,4,5,6,7,8,9,10],
#                 "nb__alpha":[0.0,0.25,0.5,0.75,1.0],
#                 }

#     grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(5,shuffle=True, random_state=0),n_jobs=-1)

#     lr_count =grid_search.fit(df["tweets"],df['class'])
#     dflr=pd.DataFrame()
#     dflr['params']=lr_count.cv_results_['params']
#     dflr['scores']=lr_count.cv_results_['mean_test_score']
#     dflr.to_csv('archivos/ES_nb_count_Dataset_results.tsv', sep='\t', index=False)

# def nbhash(df):
#     pipeline = Pipeline([('vect', HashingVectorizer()),
#                     ('nb', MultinomialNB())])
    
#     parameters={"vect__analyzer": ["word","char","char_wb"],
#                 "vect__ngram_range": [(1,1),(1,2),(2,2),(2,3),(3,3),(3,4),(4,4)],
#                 "vect__min_df":[1,2,3,4,5,6,7,8,9,10],
#                 "nb__alpha":[0.0,0.25,0.5,0.75,1.0],
#                 }

#     grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(5,shuffle=True, random_state=0),n_jobs=-1)

#     lr_count =grid_search.fit(df["tweets"],df['class'])
#     dflr=pd.DataFrame()
#     dflr['params']=lr_count.cv_results_['params']
#     dflr['scores']=lr_count.cv_results_['mean_test_score']
#     dflr.to_csv('archivos/ES_nb_hash_Dataset_results.tsv', sep='\t', index=False)

if __name__ == '__main__':
    print("Starting...")
    df = pd.read_csv("archivos/ES_Dataset.csv",sep='\t')
    dfClean = pd.read_csv("archivos/ES_CleanDataset.csv",sep='\t')
    X_train, X_test= train_test_split(df,test_size=0.20,random_state=42)
    nbtfidf(dfClean)
    print("ES_NB_Dataset+TFIDF DONE")
    #X_train, X_test= train_test_split(dfClean,test_size=0.20,random_state=42)
    #nbtfidf2(X_train,X_test)
    #print("ES_NB_CleanDataset+TFIDF DONE")



#               precision    recall  f1-score   support

#            0       0.81      0.68      0.74        19
#            1       0.75      0.86      0.80        21

#     accuracy                           0.78        40
#    macro avg       0.78      0.77      0.77        40
# weighted avg       0.78      0.78      0.77        40

# ES_NB_Dataset+TFIDF DONE

#               precision    recall  f1-score   support

#            0       0.82      0.74      0.78        19
#            1       0.78      0.86      0.82        21

#     accuracy                           0.80        40
#    macro avg       0.80      0.80      0.80        40
# weighted avg       0.80      0.80      0.80        40

# ES_NB_CleanDataset+TFIDF DONE

    
