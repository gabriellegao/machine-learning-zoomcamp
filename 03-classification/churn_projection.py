#!/usr/bin/env python
# coding: utf-8

# ## Data Preparation
# - Download the data, read it with pandas
# - Look at the data
# - Make column names and values look uniform
# - Check if all the columns read correctly
# - Check if the churn variable needs any preparation

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# In[4]:


path = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'
get_ipython().system('curl -sSL $path > "churn_projection.csv"')


# In[143]:


df = pd.read_csv('churn_projection.csv')
df.head()


# In[144]:


df.columns = df.columns.str.lower().str.replace(' ','_')


# In[145]:


# Use dtypes to filter out categorical columns 
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')


# In[146]:


df.head()


# In[147]:


# change data type from object to number, and set error cell to NaN/null
tc = pd.to_numeric(df.totalcharges, errors='coerce')


# In[148]:


# Return all rows that 'totalcharge' field contains " "(blank space)
df[tc.isnull()][['customerid','totalcharges']]


# In[149]:


df.totalcharges = pd.to_numeric(df.totalcharges, errors = 'coerce')


# In[150]:


# Fill the null cell with 0
df.totalcharges = df.totalcharges.fillna(0)


# In[151]:


# Convert yes/no to 1/0 using astype
df.churn = (df.churn == 'yes').astype(int)
df.churn


# ## Setup Validation Framework
# - Perform the train/validation/test split with Scikit-Learn

# In[152]:


from sklearn.model_selection import train_test_split


# In[153]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# In[154]:


len(df_full_train), len(df_test)


# In[155]:


df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)


# In[156]:


len(df_train), len(df_val)


# In[157]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[158]:


y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values


# In[159]:


del df_train['churn']
del df_val['churn']
del df_test['churn']


# ## EDA
# - Check missing values
# - Look at the target variable (churn)
# - Look at numerical and categorical variables

# In[160]:


df_full_train = df_full_train.reset_index(drop=True)


# In[161]:


# Cout number of records group by values
df_full_train.churn.value_counts()


# In[162]:


df_full_train.churn.value_counts(normalize=True)


# In[163]:


global_churn_rate = df_full_train.churn.mean()
round(global_churn_rate,2)


# In[164]:


numerical = ['tenure','monthlycharges','totalcharges']
numerical


# In[119]:


categorical = df_full_train.dtypes[df_full_train.dtypes=='object'].index
set(categorical) - set(numerical)


# In[165]:


categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]


# In[166]:


df_full_train[categorical].nunique()


# ## Feature Importance: Churn Rate and Risk Ratio
# Feature importance analysis (part of EDA) - identifying which features affect our target variable
# 
# - Churn rate
# - Risk ratio
# - Mutual information - later

# #### Churn Rate

# In[174]:


global_churn = df_full_train.churn.mean()
global_churn


# In[172]:


churn_female = df_full_train[df_full_train.gender == 'female'].churn.mean()
churn_female


# In[173]:


churn_male = df_full_train[df_full_train.gender == 'male'].churn.mean()
churn_male


# In[175]:


churn_partner = df_full_train[df_full_train.partner == 'yes'].churn.mean()
churn_partner


# In[176]:


churn_no_partner = df_full_train[df_full_train.partner == 'no'].churn.mean()
churn_no_partner


# In[177]:


# > 0, the group is less likely to churn
# < 0, the group is more likely to churn
global_churn - churn_female
global_churn - churn_male
global_churn - churn_partner
global_churn - churn_no_partner


# #### Risk Ratio

# In[178]:


churn_partner/global_churn


# In[179]:


churn_no_partner/global_churn


# In[180]:


# > 1, more likely to churn
# < 1, less likely to churn 


# In[ ]:


from IPython.display import display


# In[184]:


for c in categorical:
    print(c)
    df_group = df_full_train.groupby(c).churn.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn
    df_group['risk'] = df_group['mean'] / global_churn
    display(df_group)
    print()


# ## Feature Importance: Mutal Information
# Mutual information - concept from information theory, it tells us how much we can learn about one variable if we know the value of another

# In[185]:


from sklearn.metrics import mutual_info_score


# In[186]:


mutual_info_score(df_full_train.churn, df_full_train.contract)


# In[193]:


float(mutual_info_score(df_full_train.gender, df_full_train.churn))


# In[204]:


def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)


# In[210]:


# Pandas apply() is similar to RDD map()
mi = df_full_train[categorical].apply(mutual_info_churn_score)
mi.sort_values(ascending=False)


# In[367]:


df_full_train.head()


# ## Feature Importance: Correlation
# Correlation coefficient

# In[211]:


df_full_train[numerical].corrwith(df_full_train.churn)


# In[214]:


df_full_train[numerical].corrwith(df_full_train.churn).abs()


# In[212]:


df_full_train[df_full_train.tenure <= 2].churn.mean()


# In[213]:


df_full_train[(df_full_train.tenure > 2) & df_full_train.tenure <= 12].churn.mean()


# ## One-Hot Encoding
# Use Scikit-Learn to encode categorical features

# In[215]:


from sklearn.feature_extraction import DictVectorizer


# In[233]:


# Convert dataframe to dict and store each row into a dict
dicts = df_train[['gender','contract','tenure']].iloc[:100].to_dict(orient='records')


# In[234]:


# DictVectorizer can only accept dict and mapping 
# The default output is sparse matrix
dv = DictVectorizer(sparse=False)


# In[235]:


# Fit dataframe to DictVectorizer
dv.fit(dicts)


# In[236]:


# Return names of each vectorized column
dv.get_feature_names_out()


# In[237]:


# Return vectorized results
dv.transform(dicts)


# In[240]:


train_dicts = df_train[categorical + numerical].to_dict(orient='records')


# In[244]:


dv = DictVectorizer(sparse=False)


# In[245]:


dv.fit(train_dicts)


# In[246]:


dv.get_feature_names_out()


# In[251]:


list(dv.transform(train_dicts)[0])


# In[252]:


X_train = dv.fit_transform(train_dicts)
X_train.shape


# In[254]:


val_dicts = df_val[categorical+numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)


# ## Logistics Regression
# - Binary classification
# - Linear vs logistics regression

# In[255]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[256]:


z = np.linspace(-7, 7, 51)


# In[257]:


# X axis represents z (score)
# Y axis represents sigmoid(z) (probability)
plt.plot(z, sigmoid(z))


# In[258]:


def linear_regression(xi):
    result = w0
    
    for j in range(len(w)):
        result = result + xi[j] * w[j]
        
    return result


# In[259]:


def logistic_regression(xi):
    score = w0
    
    for j in range(len(w)):
        score = score + xi[j] * w[j]
        
    result = sigmoid(score)
    return result


# ## Training logistic regression with Scikit-Learn
# - Train a model with Scikit-Learn
# - Apply it to the validation dataset
# - Calculate the accuracy

# In[260]:


from sklearn.linear_model import LogisticRegression 


# In[262]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[268]:


model.coef_[0].round(3)


# In[269]:


model.intercept_[0]


# In[270]:


# Hard prediction
# Output 0 and 1
model.predict(X_train)


# In[273]:


# Soft prediction
# Output probability, right proba for 0 and left prob for 1
model.predict_proba(X_train)[:,1]


# In[276]:


# Extract probability for 1
y_pred = model.predict_proba(X_val)[:,1]


# In[280]:


# Set 0.5 as churn threshold
churn_decision = (y_pred >= 0.5)


# In[283]:


df_val[churn_decision].customerid


# In[284]:


churn_decision.astype(int)


# In[285]:


y_val


# In[287]:


(y_val == churn_decision).mean()


# In[289]:


df_pred = pd.DataFrame()
df_pred['probability'] = y_pred
df_pred['prediction'] = churn_decision.astype(int)
df_pred['actual'] = y_val


# In[291]:


df_pred['correct'] = df_pred.prediction == df_pred.actual


# In[293]:


df_pred.correct.mean()


# ## Model Interpretation
# - Look at the coefficients
# - Train a smaller model with fewer features

# In[297]:


# dv.get_feature_names_out() returns feature variable names
# model.coef_ returns weights for feature variables. 
list(zip(dv.get_feature_names_out(),model.coef_[0].round(3)))


# In[298]:


dict(zip(dv.get_feature_names_out(),model.coef_[0].round(3)))


# In[299]:


small = ['contract','tenure','monthlycharges']


# In[301]:


df_train[small].iloc[:10].to_dict(orient='records')


# In[302]:


dicts_train_small = df_train[small].to_dict(orient='records')
dicts_val_small = df_val[small].to_dict(orient='records')


# In[303]:


dv_small = DictVectorizer(sparse=False)


# In[304]:


dv_small.fit(dicts_train_small)


# In[305]:


dv_small.get_feature_names_out()


# In[307]:


X_train_small = dv_small.transform(dicts_train_small)


# In[308]:


X_val_small = dv_small.transform(dicts_val_small)


# In[309]:


model_small = LogisticRegression()


# In[310]:


model_small.fit(X_train_small, y_train)


# In[313]:


w0 = model_small.intercept_[0]
w0


# In[314]:


w = model_small.coef_[0]
w


# In[315]:


dict(zip(dv_small.get_feature_names_out(),w.round(3)))


# In[319]:


# Score
-2.47 + 0.97 + 50*0.027 + 5 * (-0.036)


# In[320]:


# _ represents the output of previous cell
# Input score into sigmoid func and output probability
sigmoid(_)


# ## Using Model

# In[324]:


dicts_full_train = df_full_train[categorical + numerical].to_dict(orient='records')


# In[325]:


dv_full_train = DictVectorizer(sparse=False)
X_full_train = dv_full_train.fit_transform(dicts_full_train)


# In[326]:


X_full_train


# In[330]:


y_full_train = df_full_train.churn.values


# In[331]:


model_full_train = LogisticRegression()


# In[332]:


model_full_train.fit(X_full_train, y_full_train)


# In[335]:


dicts_test = df_test[categorical + numerical].to_dict(orient='records')


# In[336]:


X_test = dv_full_train.transform(dicts_test)


# In[339]:


y_pred = model_full_train.predict_proba(X_test)[:,1]


# In[341]:


churn_decision = (y_pred >= 0.5)


# In[343]:


(churn_decision == y_test).mean()


# In[348]:


customer = dicts_test[10]


# In[350]:


X_small = dv_full_train.transform(customer)


# In[364]:


y_small = model_full_train.predict_proba(X_small)[0,1]
y_small


# In[354]:


y_test[10]


# In[366]:


(y_small >= 0.5).astype(int)


# In[ ]:




