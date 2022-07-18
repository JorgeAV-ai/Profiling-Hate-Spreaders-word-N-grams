import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import  TfidfVectorizer, CountVectorizer, HashingVectorizer
import pickle

def svmtfidf(df):
    pipeline = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True)),
                    ('svm', SVC())])
    
    parameters={"vect__analyzer": ["word"],
                "vect__ngram_range": [(1,1),(1,2),(2,2),(2,3),(3,3),(1,3)],
                "vect__min_df":[1,2,3,4,5,6,7,8,9,10],
                "svm__C":[1,10,100,1000],
                "svm__kernel":["linear"]}

    grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(10,shuffle=True, random_state=0),n_jobs=-1, verbose=1)

    svm_tfidf =grid_search.fit(df["tweets"],df['class'])
    dflr=pd.DataFrame()
    dflr['params']=svm_tfidf.cv_results_['params']
    dflr['scores']=svm_tfidf.cv_results_['mean_test_score']
    dflr.to_csv('archivos/ES_SVM_tfidf_Dataset_results.tsv', sep='\t', index=False)

def evalSVM(train,test):
    pipe = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True,analyzer= "word",
        ngram_range=(1,6), min_df = 0.05)), ('svm', SVC(C=100, kernel = "linear", gamma = "scale"))])

    pipe.fit(train['tweets'],train['class'])
    print(pipe.score(test['tweets'],test['class']))



def rftfidf(df):
    pipeline = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True)),
                    ('rf', RandomForestClassifier())])
    
    parameters={"vect__analyzer": ["word"],
                "vect__ngram_range": [(1,1),(1,2),(2,2)],
                "vect__min_df":[4,5,6,7,8,9,10],
                "rf__criterion":["gini", "entropy"],
                "rf__max_depth":[2,3,4,5,6],
                "rf__min_samples_split": [2,3,4,5,6,7,8,9,10]}

    grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(10,shuffle=True, random_state=0),n_jobs=-1, verbose=1)

    svm_tfidf =grid_search.fit(df["tweets"],df['class'])
    dflr=pd.DataFrame()
    dflr['params']=svm_tfidf.cv_results_['params']
    dflr['scores']=svm_tfidf.cv_results_['mean_test_score']
    dflr.to_csv('archivos/ES_RF_tfidf_Dataset_results.tsv', sep='\t', index=False)

# def lrcount(df):
#     pipeline = Pipeline([('vect', CountVectorizer()),
#                     ('lr', LogisticRegression(penalty='l2', solver='liblinear',verbose=0, random_state=5))])
    
#     parameters={"vect__analyzer": ["word","char","char_wb"],
#                 "vect__ngram_range": [(1,1),(1,2),(2,2),(2,3),(3,3),(3,4),(4,4)],
#                 "vect__min_df":[1,2,3,4,5,6,7,8,9,10,11],
#                 "lr__C":[1,10,100,1000,10000,100000]}

#     grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(5,shuffle=True, random_state=0),n_jobs=-1)

#     lr_count =grid_search.fit(df["tweets"],df['class'])
#     dflr=pd.DataFrame()
#     dflr['params']=lr_count.cv_results_['params']
#     dflr['scores']=lr_count.cv_results_['mean_test_score']
#     dflr.to_csv('archivos/ES_lr_count_CleanDataset_results.tsv', sep='\t', index=False)

# def lrhash(df):
#     pipeline = Pipeline([('vect', HashingVectorizer()),
#                     ('lr', LogisticRegression(penalty='l2', solver='liblinear',verbose=0, random_state=5))])
    
#     parameters={"vect__analyzer": ["word","char","char_wb"],
#                 "vect__ngram_range": [(1,1),(1,2),(2,2),(2,3),(3,3),(3,4),(4,4)],
#                 "vect__n_features":[10000,100000,1000000,10000000],
#                 "lr__C":[1,10,100,1000,10000,100000]}

#     grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(5,shuffle=True, random_state=0),n_jobs=-1)

#     lr_count =grid_search.fit(df["tweets"],df['class'])
#     dflr=pd.DataFrame()
#     dflr['params']=lr_count.cv_results_['params']
#     dflr['scores']=lr_count.cv_results_['mean_test_score']
#     dflr.to_csv('archivos/ES_lr_hash_CleanDataset_results.tsv', sep='\t', index=False)


if __name__ == '__main__':
    print("Starting...")
    df = pd.read_csv("archivos/ES_Dataset.csv",sep='\t')
    dfClean = pd.read_csv("archivos/ES_CleanDataset.csv",sep='\t')
    X_train, X_test = train_test_split(dfClean, test_size=0.20, random_state=42)
    #svmtfidf(dfClean)
    #evalSVM(X_train, X_test)
    rftfidf(dfClean)
