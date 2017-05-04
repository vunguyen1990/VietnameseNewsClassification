
# coding: utf-8

# In[1]:

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from sklearn import linear_model
from time import time
from sklearn.model_selection import train_test_split


# # I - Loading data

# ## 1 - Get categories of text

# In[2]:

dir_path = os.path.join(os.getcwd(), 'vnexpress')
categories = list()

data = list()
for directory in os.listdir(dir_path):
#     print(directory)
    if '.' not in directory:
        list_file_path = os.path.join(dir_path, directory)
        count = 0
        for file_name in os.listdir(list_file_path):
            data_dict = dict()
            data_dict['category'] = directory
            file_path = os.path.join(list_file_path, file_name)
            file = open(file_path,'r')
            data_dict['data'] = file.read()
            data.append(data_dict)


# ## sample 13000 items to training and testing

# In[3]:

data_df = pd.DataFrame(data)
sample_df = data_df.sample(13000)


# In[4]:

training_df = sample_df[:10000]
training_data = training_df.data
training_label = training_df.category
testing_df = sample_df[10000:]
testing_data = sample_df.data
testing_label = sample_df.category


# ### Using all dataset

# In[5]:

train, test = train_test_split(data_df, test_size = 0.3)


# In[6]:

# training_df = sample_df[:10000]
training_data = train.data
training_label = train.category
# testing_df = sample_df[10000:]
testing_data = test.data
testing_label = test.category


# ## Fitting tf-idf model

# In[7]:

t0 = time()
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
vectorizer.fit(data_df.data)
print("tf-idf learning time:", time() - t0)


# ## Extracting features for training and testing data

# In[8]:

t0 = time()
training_matrix = vectorizer.transform(training_data)
testing_matrix = vectorizer.transform(testing_data)
print("feature extraction time of training and testing dataset:", time() - t0)


# In[9]:

training_vector = training_matrix.toarray()
testing_vector = testing_matrix.toarray()


# ## Classification by Logistic Regression of Scikit Learn

# In[ ]:

t0 = time()
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(training_vector,training_label)
print('Logistic Regression time:',time()-t0)


# In[ ]:

t0 = time()
predicted_result = logreg.predict(testing_vector)
print("testing time:", time()-t0)


# In[ ]:

from sklearn import svm,metrics,model_selection
print(metrics.classification_report(testing_label, predicted_result))

