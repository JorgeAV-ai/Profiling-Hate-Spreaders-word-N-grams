import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import  TfidfVectorizer, CountVectorizer, HashingVectorizer
import pickle
import time

def knntfidf(df):
    pipeline = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True)),
                    ('knn', KNeighborsClassifier(n_jobs=-1))])
    
    parameters={"vect__analyzer": ["word"],
                "vect__ngram_range": [(1,1),(1,2),(2,2)],
                "vect__min_df":[4,5,6,7,8,9,10],
                "knn__weights":["uniform","distance"],
                "knn__metric":["euclidean","manhattan","minkowski"],
                "knn__n_neighbors":[1,3,5],
                "knn__leaf_size":[10,20,40],
                "knn__p":[1,2]
                }

    grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(10,shuffle=True, random_state=0),verbose=2,n_jobs=-1)

    lr_tfidf =grid_search.fit(df["tweets"],df['class'])
    dfknn=pd.DataFrame()
    dfknn['params']=lr_tfidf.cv_results_['params']
    dfknn['scores']=lr_tfidf.cv_results_['mean_test_score']
    dfknn.to_csv('archivos/ES_KNN_tfidf_Dataset_results.tsv', sep='\t', index=False)

    # given the best hyperparameters
    #y_true, y_pred = X_test["class"], lr_tfidf.predict(X_test["tweets"])
    #print(classification_report(y_true, y_pred))



def knntfidf2(df,X_test):
    pipeline = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True)),
                    ('knn', KNeighborsClassifier(n_jobs=-1))])
    
    parameters={"vect__analyzer": ["word"],
                "vect__ngram_range": [(2,2),(2,3),(3,3)],
                "vect__min_df":[1,2,3,4,5],
                "knn__weights":["uniform","distance"],
                "knn__metric":["euclidean","manhattan","minkowski"],
                "knn__n_neighbors":[3,5,10,20],
                "knn__leaf_size":[20,40,1],
                "knn__p":[1,2]
                }

    grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(5,shuffle=True, random_state=0),n_jobs=-1)

    lr_tfidf =grid_search.fit(df["tweets"],df['class'])
    dfknn=pd.DataFrame()
    dfknn['params']=lr_tfidf.cv_results_['params']
    dfknn['scores']=lr_tfidf.cv_results_['mean_test_score']
    dfknn.to_csv('archivos/ES_KNN_tfidf_CleanDataset_results.tsv', sep='\t', index=False)

    # given the best hyperparameters
    y_true, y_pred = X_test["class"], lr_tfidf.predict(X_test["tweets"])
    print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    print("Starting...")
    df = pd.read_csv("archivos/ES_Dataset.csv",sep='\t')
    dfClean = pd.read_csv("archivos/ES_CleanDataset.csv",sep='\t')
    time1 = time.time()
    print(time1)
   # X_train, X_test= train_test_split(df,test_size=0.20,random_state=42)
    knntfidf(dfClean)
    print("ES_KNN_Dataset+TFIDF DONE")
    #X_train, X_test= train_test_split(dfClean,test_size=0.20,random_state=42)
    #knntfidf2(X_train,X_test)
    #print("ES_KNN_CleanDataset+TFIDF DONE")
    print( time.time()- time1 )




# Resultados  Not cleaned         
                # precision    recall  f1-score   support

#            0       0.82      0.74      0.78        19
#            1       0.78      0.86      0.82        21

#     accuracy                           0.80        40
#    macro avg       0.80      0.80      0.80        40
# weighted avg       0.80      0.80      0.80        40

# ES_KNN_Dataset+TFIDF DONE


# Resultados  cleaned
#               precision    recall  f1-score   support

#            0       0.77      0.89      0.83        19
#            1       0.89      0.76      0.82        21

#     accuracy                           0.82        40
#    macro avg       0.83      0.83      0.82        40
# weighted avg       0.83      0.82      0.82        40

# ES_KNN_CleanDataset+TFIDF DONE
