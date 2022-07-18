# Voting Classifier

import pandas as pd
from numpy import mean
from numpy import std
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


def models_voting():
    estimators = [
        ('lr',make_pipeline(TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True,analyzer='word',min_df=9,ngram_range=(1,2)),LogisticRegression(C=1000,penalty='l2', solver='liblinear',verbose=0, random_state=0))),
        ('nb',make_pipeline(TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True,analyzer='word',min_df=5,ngram_range=(1,2)),MultinomialNB(alpha=0.25,fit_prior=True))),
        ('knn',make_pipeline(TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True,analyzer='word',min_df=5,ngram_range=(2,2)),KNeighborsClassifier(n_jobs=-1,leaf_size=20,metric='euclidean',n_neighbors=10,p=1,weights='distance'))),
        ('rf',make_pipeline(TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True,analyzer='word',min_df=0.05,ngram_range=(1,6)),RandomForestClassifier(criterion='gini',max_depth=3,min_samples_split=6,random_state=0))  ),
        # ('svm',make_pipeline(TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True,analyzer= "word",ngram_range=(1,6), min_df = 0.05), SVC(C=100, kernel = "linear", gamma = "scale",probability=True)))
    ]

    ensemble = VotingClassifier(estimators=estimators,voting='hard')
    return ensemble

def get_models():
    models= {}
    models['LR'] = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True,analyzer='word',min_df=9,ngram_range=(1,2))),('lr', LogisticRegression(C=1000,penalty='l2', solver='liblinear',verbose=0, random_state=0))])
    models['NB'] = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True,analyzer='word',min_df=5,ngram_range=(1,2))),('nb', MultinomialNB(alpha=0.25,fit_prior=True))])
    # models['SGD'] =  
    models['KNN'] = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True,analyzer='word',min_df=5,ngram_range=(2,2))),('knn', KNeighborsClassifier(n_jobs=-1,leaf_size=20,metric='euclidean',n_neighbors=10,p=1,weights='distance'))])
    models['RF'] = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True,analyzer='word',min_df=0.05,ngram_range=(1,6))),('rf', RandomForestClassifier(criterion='gini',max_depth=3,min_samples_split=6,random_state=0))])
    # models['SVC'] = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True,analyzer= "word",ngram_range=(1,6), min_df = 0.05)), ('svm', SVC(C=100, kernel = "linear", gamma = "scale",probability=True))])
    models['soft_voting'] = models_voting()
    return models

def evaluate_model(model,X,y):
    model_fitted = model.fit(X["tweets"],X['class'])
    y_true, y_pred = y["class"], model_fitted.predict(y["tweets"])
    print(y_pred)
    res = classification_report(y_true, y_pred)
    # res = cross_val_score(model,X["tweets"],y["tweets"],cv=StratifiedKFold(5,shuffle=True, random_state=0))
    return res



if __name__ == '__main__':
    print("Starting...")
    dfClean = pd.read_csv("archivos/ES_CleanDataset.csv",sep='\t')
    X_train, X_test= train_test_split(dfClean,test_size=0.20,random_state=42)
    results, names = list(),list()
    print(X_test['class'])
    models = get_models()
    for name, model in models.items():
        scores = evaluate_model(model,X_train,X_test)
        # results.append(scores)
        # names.append(name)
        print(name,scores)
        # print(name,mean(scores),std(scores))



# RL               precision    recall  f1-score   support

#            0       0.83      0.79      0.81        19
#            1       0.82      0.86      0.84        21

#     accuracy                           0.82        40
#    macro avg       0.83      0.82      0.82        40
# weighted avg       0.83      0.82      0.82        40

# [1 1 1 0 0 1 1 0 1 0 0 1 1 1 0 1 0 0 0 0 1 1 1 1 1 0 0 0 1 0 1 1 0 0 1 1 1
#  1 1 0]
# NB               precision    recall  f1-score   support

#            0       0.82      0.74      0.78        19
#            1       0.78      0.86      0.82        21

#     accuracy                           0.80        40
#    macro avg       0.80      0.80      0.80        40
# weighted avg       0.80      0.80      0.80        40

# [1 1 1 0 0 1 1 0 1 0 0 1 1 1 0 1 0 0 0 0 0 0 1 0 1 0 0 0 1 0 1 0 0 0 1 1 1
#  1 0 0]
# KNN               precision    recall  f1-score   support

#            0       0.77      0.89      0.83        19
#            1       0.89      0.76      0.82        21

#     accuracy                           0.82        40
#    macro avg       0.83      0.83      0.82        40
# weighted avg       0.83      0.82      0.82        40

# [1 1 1 0 0 0 1 0 1 0 0 1 1 1 0 1 0 0 0 0 1 1 1 1 1 0 0 0 1 0 1 0 0 0 1 1 1
#  1 0 0]
# RF               precision    recall  f1-score   support

#            0       0.80      0.84      0.82        19
#            1       0.85      0.81      0.83        21

#     accuracy                           0.82        40
#    macro avg       0.82      0.83      0.82        40
# weighted avg       0.83      0.82      0.83        40

# [1 1 1 0 0 1 1 0 1 0 0 1 1 1 0 1 0 0 0 0 0 1 1 1 1 0 0 0 1 0 1 0 0 0 1 1 1
#  1 0 0]
# hard_voting               precision    recall  f1-score   support

#            0       0.85      0.89      0.87        19
#            1       0.90      0.86      0.88        21

#     accuracy                           0.88        40
#    macro avg       0.88      0.88      0.87        40
# weighted avg       0.88      0.88      0.88        40
