import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import numpy as np
from sklearn import linear_model
from time import time
from sklearn.model_selection import train_test_split
from sklearn import svm,metrics,model_selection

def logistic_regression(sample_size,data_df):

    result = dict()
    result_time = dict()

    sample_df = data_df.sample(sample_size)
    train, test = train_test_split(sample_df, test_size = 0.3)
    print("training:%d -  testing: %d \n"%(len(train),len(test)))


    # In[25]:

    training_data = train.data
    training_label = train.category
    testing_data = test.data
    testing_label = test.category


    # ## Fitting tf-idf model

    # In[26]:

    t0 = time()
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.75,binary= False,smooth_idf = False,norm='l2')
    vectorizer.fit(training_data)
    print("tf-idf learning time:", time() - t0)
    result_time['tf_idf'] = time() - t0

    # In[27]:

    result['vectorizer'] = vectorizer


    # ## Extracting features for training and testing data

    # In[28]:

    t0 = time()
    training_matrix = vectorizer.transform(training_data)
    testing_matrix = vectorizer.transform(testing_data)
    print("feature extraction time of training and testing dataset:", time() - t0)
    result_time['feature_extraction'] = time() - t0

    # In[29]:

    print('training',training_matrix.shape)
    print('testing',testing_matrix.shape)
    shape = training_matrix.shape


    result['vector_size'] = shape[1]


    # ## Classification by Logistic Regression of Scikit Learn

    # In[31]:

    t0 = time()
    logreg = linear_model.LogisticRegression(C=100000.0)
    logreg.fit(training_matrix,training_label)
    print('Logistic Regression time:',time()-t0)
    result_time['training_classifer'] = time() - t0


    # In[32]:

    t0 = time()
    predicted_result = logreg.predict(testing_matrix)
    print("testing time:", time()-t0)
    result_time['test_classifier'] = time() - t0

    print(metrics.classification_report(testing_label, predicted_result))
    result['time_consume'] = result_time
    precision = np.mean(metrics.precision_recall_fscore_support(testing_label, predicted_result)[0])
    recall = np.mean(metrics.precision_recall_fscore_support(testing_label, predicted_result)[1])
    fscore = np.mean(metrics.precision_recall_fscore_support(testing_label, predicted_result)[2])
    result['precision'] = precision
    result['recall'] = recall
    result['fscore'] = fscore
    return result



