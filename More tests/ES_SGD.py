import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import  TfidfVectorizer, CountVectorizer, HashingVectorizer
import pickle

def sgdtfidf(df,X_test):
    pipeline = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True)),
                    ('sgd', SGDClassifier(max_iter=100,n_jobs=-1,verbose=0, random_state=5))])
    
    parameters={"vect__analyzer": ["word"],
                "vect__ngram_range": [(1,1),(1,2),(2,2)],
                "vect__min_df":[2,3,4,5,6,7,8,9,10],
                "sgd__alpha":[0.0001,0.001,0.01],
                "sgd__loss":["hinge","log","squared_hinge","modified_huber","perceptron","modified_huber"]
                }

    grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(10,shuffle=True, random_state=0),n_jobs=-1)

    sgd_tfidf =grid_search.fit(df["tweets"],df['class'])
    dfsgd=pd.DataFrame()
    dfsgd['params']=sgd_tfidf.cv_results_['params']
    dfsgd['scores']=sgd_tfidf.cv_results_['mean_test_score']
    dfsgd.to_csv('archivos/ES_SGD_tfidf_Dataset_results.tsv', sep='\t', index=False)
    
    # given the best hyperparameters
    y_true, y_pred = X_test["class"], sgd_tfidf.predict(X_test["tweets"])
    print(classification_report(y_true, y_pred))

def sgdtfidf2(df,X_test):
    pipeline = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True)),
                    ('sgd', SGDClassifier(max_iter=100,n_jobs=-1,verbose=0, random_state=5))])
    
    parameters={"vect__analyzer": ["word"],
                "vect__ngram_range": [(1,1),(1,2),(2,2)],
                "vect__min_df":[2,3,4,5,6,7,8,9,10],
                "sgd__alpha":[0.0001,0.001,0.01],
                "sgd__loss":["hinge","log","squared_hinge","modified_huber","perceptron","modified_huber"]
                }

    grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(10,shuffle=True, random_state=0),n_jobs=-1)

    sgd_tfidf =grid_search.fit(df["tweets"],df['class'])
    dfsgd=pd.DataFrame()
    dfsgd['params']=sgd_tfidf.cv_results_['params']
    dfsgd['scores']=sgd_tfidf.cv_results_['mean_test_score']
    dfsgd.to_csv('archivos/ES_SGD_tfidf_CleanDataset_result.tsv', sep='\t', index=False)
    
    # given the best hyperparameters
    y_true, y_pred = X_test["class"], sgd_tfidf.predict(X_test["tweets"])
    print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    print("Starting...")
    df = pd.read_csv("archivos/ES_Dataset.csv",sep='\t')
    dfClean = pd.read_csv("archivos/ES_CleanDataset.csv",sep='\t')
    X_train, X_test= train_test_split(df,test_size=0.20,random_state=42)
    # print(X_train.groupby(['class']).count())
    # print(X_test.groupby(['class']).count())
    
    # sgdtfidf(X_train,X_test)
    # print("ES_SGD_Dataset+TFIDF DONE")
    # X_train, X_test= train_test_split(dfClean,test_size=0.20,random_state=42)
    # sgdtfidf2(X_train,X_test)
    # print("ES_SGD_CleanDataset+TFIDF DONE")


#                   precision    recall  f1-score   support

#            0       0.79      0.79      0.79        19
#            1       0.81      0.81      0.81        21

#     accuracy                           0.80        40
#    macro avg       0.80      0.80      0.80        40
# weighted avg       0.80      0.80      0.80        40

# ES_SGD_Dataset+TFIDF DONE

#               precision    recall  f1-score   support

#            0       0.76      0.68      0.72        19
#            1       0.74      0.81      0.77        21

#     accuracy                           0.75        40
#    macro avg       0.75      0.75      0.75        40
# weighted avg       0.75      0.75      0.75        40

# ES_SGD_CleanDataset+TFIDF DONE
