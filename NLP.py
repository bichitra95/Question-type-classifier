
# coding: utf-8

# # Question type classification problem

# This problem is tackeled using the concept of bag of words commonly used in NLP as there is very less words in each question. After bag of words, we used Term frequency and Inverse Document frequency to to give weightage to each words found in a sentence and the training set. After that we have used different machine learning techniques and identified that Randomforest classifier as the suitable algorithm to use. And obtain more that 92% of F1 score. 

# In[1]:

import pandas as pd
import numpy as np


# In[2]:

train=pd.read_csv('LabelledData (1).txt',sep=',,,',names=['question','label'],engine='python')


# ### 1. Exploratory Data Analysis

# In[3]:

train.head()


# In[4]:

#Removing unwanted spaces from both the ends

train['label']=train['label'].apply(lambda x: x.strip())
train['question']=train['question'].apply(lambda x: x.strip())


# In[5]:

train.describe()# shows duplicate questions as count and unique are different for question


# In[6]:

train.drop_duplicates(inplace=True)


# In[7]:

train['label'].value_counts()


# In[8]:

train.groupby(by='label').describe()


# In[9]:

train.describe()


# ### 2. Data processing

# In[10]:

import string
import nltk


# In[11]:

#filtering the punctuations from the sentence and tokenizing the sentence into words
def text_process(question):
    noponc=[word for word in question if word not in string.punctuation]
    noponc=''.join(noponc).strip()
    #return [word for word in noponc.split() if word.lower() in stopwords.words('english')]
    return noponc.split()


# #### Exploring for vectorizing questions in to bow using predefined analyzer. 

# In[12]:

from sklearn.feature_extraction.text import CountVectorizer


# In[13]:

bow_trasformer=CountVectorizer(analyzer=text_process).fit(train['question'])


# In[14]:

print(len(bow_trasformer.vocabulary_))


# In[15]:

messages_bow=bow_trasformer.transform(train['question'])


# In[16]:

messages_bow.shape


# ### Exploring for transforming the bag of words table into Tfidf.

# In[17]:

from sklearn.feature_extraction.text import TfidfTransformer


# In[18]:

tfidf_transformer=TfidfTransformer().fit(messages_bow)


# In[19]:

message_tfidf=tfidf_transformer.transform(messages_bow)


# ## 3. Loading different learning algorithms 

# In[20]:

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
classifier=[MultinomialNB(),LogisticRegression(),RandomForestClassifier()]


# In[ ]:




# In[ ]:




# ### 4. Splitting the data set into train test 

# In[21]:

from sklearn.model_selection import train_test_split


# In[22]:

qn_train,qn_test,label_train,label_test=train_test_split(train['question'],train['label'],test_size=0.3)


# ### 5. Creating a pipeline of process for training and prediction
# 
# 1. We have vectorized the BOW using the defined function text_process
# 2. Then generate TFIDF of BOW
# 3. Then impliment a perticular classifier 

# In[23]:

from sklearn.pipeline import Pipeline


# In[24]:

def complete_process(classifier):
    pipeline=Pipeline([
        ('bow',CountVectorizer(analyzer=text_process)),
        ('tfidf',TfidfTransformer()),
        ('classifier',classifier)
    ])
    pipeline.fit(qn_train,label_train)
    return pipeline.predict(qn_test)


# ## 6. Checking confusion matrix and derived F1 score for individual classifier
#  And find that Random forest classifier has F1 score above 92 which is comparatively high

# In[25]:

from sklearn.metrics import classification_report, confusion_matrix


# In[42]:

i=0

for classify in classifier:
    predictions=complete_process(classify)
    print(classifier[i])
    print('\n')
    print(pd.DataFrame(confusion_matrix(label_test,predictions,labels=['affirmation','unknown','what','when','who']),columns=['affirmation','unknown','what','when','who'],index=['affirmation','unknown','what','when','who']))    
    print('\n')
    print(classification_report(label_test,predictions))
    i=i+1

