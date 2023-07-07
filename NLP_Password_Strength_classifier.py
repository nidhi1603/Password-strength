#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# #### REading dataset

# In[3]:


data=pd.read_csv('E:\End-2-end Projects\Password_Classifier/data.csv',error_bad_lines=False)
data.head()


# In[4]:


data['strength'].unique()


# In[ ]:





# #### code to check all the missing values in my dataset

# In[5]:


data.isna().sum()


# In[6]:


data[data['password'].isnull()]


# In[7]:


data.dropna(inplace=True)


# In[8]:


data.isnull().sum()


# In[9]:


sns.countplot(data['strength'])


# In[10]:


password_tuple=np.array(data)


# In[11]:


password_tuple


# In[ ]:





# #### shuffling randomly for robustness

# In[12]:


import random
random.shuffle(password_tuple)


# In[13]:


x=[labels[0] for labels in password_tuple]
y=[labels[1] for labels in password_tuple]


# In[14]:


x


# In[ ]:





# #### create a custom function to split input into characters of list

# In[15]:


def word_divide_char(inputs):
    character=[]
    for i in inputs:
        character.append(i)
    return character


# In[16]:


word_divide_char('kzde5577')


# In[ ]:





# #### import TF-IDF vectorizer to convert String data into numerical data

# In[17]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[18]:


vectorizer=TfidfVectorizer(tokenizer=word_divide_char)


# #### apply TF-IDF vectorizer on data

# In[19]:


X=vectorizer.fit_transform(x)


# In[20]:


X.shape


# In[21]:


vectorizer.get_feature_names()


# In[22]:


first_document_vector=X[0]
first_document_vector


# In[23]:


first_document_vector.T.todense()


# In[26]:


df=pd.DataFrame(first_document_vector.T.todense(),index=vectorizer.get_feature_names(),columns=['TF-IDF'])
df.sort_values(by=['TF-IDF'],ascending=False)


# In[ ]:





# #### split data into train & test
#     train---> To learn the relationship within data, 
#     test-->  To do predictions, and this testing data will be unseen to my model

# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)


# In[29]:


X_train.shape


# In[30]:


from sklearn.linear_model import LogisticRegression


# In[ ]:





# #### Apply Logistic on data as use-cas is Classification

# In[31]:


clf=LogisticRegression(random_state=0,multi_class='multinomial')


# In[32]:


clf.fit(X_train,y_train)


# In[ ]:





# #### doing prediction for specific custom data

# In[33]:


dt=np.array(['%@123abcd'])
pred=vectorizer.transform(dt)
clf.predict(pred)


# In[ ]:





# #### doing prediction on X-Test data

# In[34]:


y_pred=clf.predict(X_test)
y_pred


# In[ ]:





# #### check Accuracy of your model using confusion_matrix,accuracy_score

# In[35]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[36]:


cm=confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))


# In[ ]:





# ##### create report of your model

# In[38]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:





# In[ ]:




