#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df=pd.read_csv('Combined_News_DJIA.csv',encoding="ISO-8859-1")


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


train = df[df['Date']<'20150101']
test = df[df['Date']<'20141231']


# In[7]:


data=train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

list1=[i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns=new_Index
data.head()


# In[10]:


for index in new_Index:
       data[index]=data[index].str.lower()
data.head()


# In[11]:


' '.join(str(x) for x in data.iloc[1,0:25])


# In[12]:


headlines=[]
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))


# In[13]:


headlines[0]


# In[14]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[15]:


countvector=CountVectorizer(ngram_range=(2,2))
traindataset=countvector.fit_transform(headlines)


# In[16]:


traindataset[0]


# In[21]:


randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])


# In[22]:


test_transform=[]
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)


# In[24]:


predictions


# In[23]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[27]:


test.index


# In[31]:


test.loc[1601,:]


# In[32]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[34]:


matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)


# In[49]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[51]:


TfidfVector=TfidfVectorizer(ngram_range=(2,2))
traindataset=TfidfVector.fit_transform(headlines)


# In[52]:


randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])


# In[54]:


test_transform=[]
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = TfidfVector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)


# In[55]:


predictions


# In[56]:


matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)


# In[ ]:




