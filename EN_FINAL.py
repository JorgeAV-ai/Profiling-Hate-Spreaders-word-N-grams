import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import  TfidfVectorizer, CountVectorizer, HashingVectorizer
import pickle


#TF-IDF with LogisticRegression
def lrtfidf(df):
    pipeline = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True)),
                    ('lr', LogisticRegression(penalty='l2', solver='liblinear',verbose=0, random_state=0))])
    
    parameters={"vect__analyzer": ["word"],
                "vect__ngram_range": [(1,1),(1,2),(2,2),(2,3),(3,3),(3,4),(4,4)],
                "vect__min_df":[1,2,3,4,5,6,7,8,9,10,11],
                "lr__C":[1,10,100,1000,10000,100000]}


    grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(5,shuffle=True, random_state=0),n_jobs=-1,verbose=2)

    lr_tfidf =grid_search.fit(df["tweets"],df['class'])
    dflr=pd.DataFrame()
    dflr['params']=lr_tfidf.cv_results_['params']
    dflr['scores']=lr_tfidf.cv_results_['mean_test_score']
    dflr.to_csv('archivos/EN_LR_tfidf_Dataset_results.tsv', sep='\t', index=False)

#TF-IDF with KNN
def knntfidf(df):
    pipeline = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True)),
                    ('knn', KNeighborsClassifier(n_jobs=-1))])
    
    parameters={"vect__analyzer": ["word"],
                "vect__ngram_range": [(1,2),(2,2),(2,3),(3,3)],
                "vect__min_df":[5,6,7,8,9,10],
                "knn__weights":["uniform","distance"],
                "knn__metric":["euclidean","manhattan","minkowski"],
                "knn__n_neighbors":[3,5,10],
                "knn__leaf_size":[20,40],
                "knn__p":[1]
                }

    grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(5,shuffle=True, random_state=0),n_jobs=-1)

    lr_tfidf =grid_search.fit(df["tweets"],df['class'])
    dfknn=pd.DataFrame()
    dfknn['params']=lr_tfidf.cv_results_['params']
    dfknn['scores']=lr_tfidf.cv_results_['mean_test_score']
    dfknn.to_csv('archivos/EN_KNN_tfidf_Dataset_results.tsv', sep='\t', index=False)


#TF-IDF with NaiveBayes
def nbtfidf(df):
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
    dflr.to_csv('archivos/EN_NB_tfidf_Dataset_results.tsv', sep='\t', index=False)

#TF-IDF with LogisticRegression
def sgdtfidf(df):
    pipeline = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True)),
                    ('sgd', SGDClassifier(max_iter=100,n_jobs=-1,verbose=0, random_state=5))])
    
    parameters={"vect__analyzer": ["word"],
                "vect__ngram_range": [(1,2)],
                "vect__min_df":[2,3,4,5,6,7,8,9,10],
                "sgd__alpha":[0.0001,0.001,0.01],
                "sgd__loss":["hinge","log","squared_hinge","modified_huber","perceptron","modified_huber"]
                }

    grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(10,shuffle=True, random_state=0),n_jobs=-1,verbose=2)

    sgd_tfidf =grid_search.fit(df["tweets"],df['class'])
    dfsgd=pd.DataFrame()
    dfsgd['params']=sgd_tfidf.cv_results_['params']
    dfsgd['scores']=sgd_tfidf.cv_results_['mean_test_score']
    dfsgd.to_csv('archivos/EN_SGD_3_tfidf_CleanDataset_results.tsv', sep='\t', index=False) 

#TF-IDF with SupportVectorMachine
def svmtfidf(df):
    pipeline = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True)),
                    ('svm', SVC())])
    
    parameters={"vect__analyzer": ["word"],
                "vect__ngram_range": [(1,1),(1,2),(2,2),(2,3),(3,3),(1,3)],
                "vect__min_df":[5,6,7,8,9,10,11],
                "vect__max_df": [0.95],
                "svm__C":[0.1,1,100,10000],
                "svm__kernel":["linear"]}

    grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(5,shuffle=True, random_state=0),n_jobs=-1, verbose=2)

    svm_tfidf =grid_search.fit(df["tweets"],df['class'])
    dflr=pd.DataFrame()
    dflr['params']=svm_tfidf.cv_results_['params']
    dflr['scores']=svm_tfidf.cv_results_['mean_test_score']
    dflr.to_csv('archivos/EN_SVM_jose_tfidf_CleanDataset_results.tsv', sep='\t', index=False)


#TF-IDF with RandomForest
def rftfidf(df):
    pipeline = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True)),
                    ('rf', RandomForestClassifier())])
    
    parameters={"vect__analyzer": ["word"],
                "vect__ngram_range": [(1,1),(1,2),(2,2),(2,3),(3,3),(1,3)],
                "vect__min_df":[0.001,0.005,0.01,0.05,0.10],
                "rf__criterion":["gini", "entropy"],
                "rf__max_depth":[2,3,4,5,6],
                "rf__min_samples_split": [2,4,5,6,8,10]}

    grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(5,shuffle=True, random_state=0),n_jobs=-1, verbose=2)

    svm_tfidf =grid_search.fit(df["tweets"],df['class'])
    dflr=pd.DataFrame()
    dflr['params']=svm_tfidf.cv_results_['params']
    dflr['scores']=svm_tfidf.cv_results_['mean_test_score']
    dflr.to_csv('archivos/EN_RF_tfidf_Dataset_results.tsv', sep='\t', index=False)



if __name__ == '__main__':
    print("Starting...")
    df = pd.read_csv("archivos/EN_Dataset.csv",sep='\t')
    dfClean = pd.read_csv("archivos/EN_CleanDataset.csv",sep='\t')
    cv=StratifiedKFold(5,shuffle=True, random_state=0)
    svmtfidf(dfClean)
    print("EN_SVM_CleanDataset+TFIDF DONE")

