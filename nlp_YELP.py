#!/usr/bin/env python
# coding: utf-8

# In[116]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[118]:


yelp = pd.read_csv('yelp.csv')


# In[119]:


yelp.head()


# In[120]:


yelp.describe()


# In[121]:


yelp.info()


# In[122]:


yelp['text length'] = yelp['text'].apply(len)


# <b style="color:black;">EDA</b>
# 

# In[124]:


g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length',bins=40)


# In[125]:


sns.boxplot(x='stars',y='text length',data=yelp)


# In[126]:


sns.countplot(x='stars',data=yelp)


# In[128]:


stars = yelp.groupby('stars').mean(numeric_only=True)


# In[129]:


stars


# In[134]:


stars.corr()


# In[133]:


stars.corr().plot()


# <b style="color:black;">NLP</b>

# In[141]:


yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]


# In[142]:


X = yelp_class['text']
y = yelp_class['stars']


# In[143]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# In[144]:


X = cv.fit_transform(X)


# In[145]:


from sklearn.model_selection import train_test_split


# In[146]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


# In[147]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[148]:


nb.fit(X_train,y_train)


# In[149]:


predictions = nb.predict(X_test)


# In[150]:


from sklearn.metrics import confusion_matrix,classification_report


# <b style="color:black;">Model Evaluation</b>

# In[155]:


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# In[ ]:




