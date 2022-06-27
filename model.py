#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


data=pd.read_csv('Bengaluru_House_Data.csv')


# In[5]:


data.head()


# In[6]:


data.shape


# In[7]:


data.info()


# In[8]:


for column in data.columns:
    print(data[column].value_counts())
    print('*'*20)


# In[9]:


data.isna().sum()


# In[10]:


data.drop(columns=['area_type','availability','society','balcony'], inplace=True)


# In[11]:


data.info()


# In[12]:


data.describe()


# In[13]:


data.info()


# In[14]:


data['location'].value_counts()


# In[15]:


data['location']=data['location'].fillna('Sarjapur Road')


# In[16]:


data.info()


# In[17]:


data['size'].value_counts()


# In[18]:


data['size']=data['size'].fillna('2 BHK')


# In[19]:


data.info()


# In[20]:


data['bath'].value_counts()


# In[21]:


data.isna().sum()


# In[22]:


data['bath'].median()


# In[23]:


data['bath']=data['bath'].fillna(data['bath'].median())


# In[24]:


data.info()


# In[25]:


data['bhk']=data['size'].str.split().str.get(0).astype(int)


# In[26]:


data['bhk'].info()


# In[27]:


data[data.bhk > 20]


# In[28]:


data['total_sqft'].unique()


# In[29]:


def convertRange(x):
    
    temp=x.split('-')
    if len(temp) == 2:
        return (float(temp[0]) + float(temp[1]))/2
    try:
        return float(x)
    except:
        return None


# In[30]:


data['total_sqft']=data['total_sqft'].apply(convertRange)


# In[31]:


data['total_sqft'].tail()


# In[32]:


data.head()


# In[33]:


data['price_per_sqft']=data['price']*100000/data['total_sqft']


# In[34]:


data.head()


# In[35]:


data.describe()


# In[36]:


data['location'].value_counts()


# In[37]:


data['location']=data['location'].apply(lambda x: x.strip())
location_count=data['location'].value_counts()


# In[38]:


location_count


# In[39]:


location_count_less_10 = location_count[location_count <=10]
location_count_less_10


# In[40]:


data['location']=data['location'].apply(lambda x: 'other' if x in location_count_less_10 else x)


# In[41]:


data[data['location'] == 'other']


# In[42]:


data['location'].value_counts()


# ## removing and treating outliers

# In[43]:


data.describe()


# In[44]:


(data['total_sqft']/data['bhk']).describe()


# In[45]:


data=data[((data['total_sqft']/data['bhk']) >=300)]
data.describe()


# In[46]:


data.shape


# In[47]:


def remove_outliers_sqft(df):
    df_output=pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        
        st = np.std(subdf.price_per_sqft)
        gen_df=subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        
        df_output= pd.concat([df_output,gen_df], ignore_index=True)
    return df_output

data=remove_outliers_sqft(data)
data.describe()


# In[48]:


def bhk_outlier_remover(df):
    exclude_indices=np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats={}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
            
    return df.drop(exclude_indices,axis='index')


# In[49]:


data=bhk_outlier_remover(data)


# In[50]:


data


# In[51]:


data.drop(columns=['size','price_per_sqft'], inplace=True)


# ## Cleaned Data

# In[52]:


data.head()


# In[53]:


data.to_csv('Cleaned_data.csv')


# In[54]:


X=data.drop(columns=['price'])
y=data['price']


# In[55]:


data['bhk'].describe()


# In[56]:


data['bath'].describe()


# ## Importing model training libraries

# In[57]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# In[59]:


print(X_train.shape)
print(X_test.shape)


# ## Applying Linear Regression

# In[60]:


column_trans = make_column_transformer((OneHotEncoder(sparse=False), ['location']), remainder='passthrough')


# In[61]:


scaler=StandardScaler()


# In[62]:


lr = LinearRegression(normalize=True)


# In[63]:


pipe=make_pipeline(column_trans, scaler, lr)


# In[64]:


pipe.fit(X_train, y_train)


# In[65]:


y_pred_lr = pipe.predict(X_test)


# In[66]:


r2_score(y_test, y_pred_lr)


# ## Applying Lasso

# In[67]:


lasso= Lasso()


# In[68]:


pipe = make_pipeline(column_trans, scaler, lasso)


# In[69]:


pipe.fit(X_train, y_train)


# In[70]:


y_pred_lasso = pipe.predict(X_test)
r2_score(y_test, y_pred_lasso)


# ## Applying Ridge

# In[71]:


ridge=Ridge()


# In[72]:


pipe = make_pipeline(column_trans, scaler, ridge)


# In[73]:


pipe.fit(X_train, y_train)


# In[74]:


y_pred_ridge=pipe.predict(X_test)
r2_score(y_test, y_pred_ridge)


# In[77]:


print("No Regularization: ", r2_score(y_test, y_pred_lr))
print("Lasso: ", r2_score(y_test, y_pred_lasso))
print("Ridge: ", r2_score(y_test, y_pred_ridge))


# In[78]:


import pickle


# In[79]:


pickle.dump(pipe, open('RidgeModel.pkl', 'wb'))


# In[ ]:




