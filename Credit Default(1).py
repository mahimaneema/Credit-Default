#!/usr/bin/env python
# coding: utf-8

# # Credit Default Model

# Called for a mission(Data Modelling), Now time to reach at destination so I better Load the necessary equipment (Dataset) for the mission.
# 
# 

# In[33]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
from matplotlib import pyplot
from pylab import rcParams
import seaborn as sns

import scipy 

import statsmodels.api as sm

import sklearn

from sklearn import preprocessing


import warnings
warnings.filterwarnings("ignore")


# In[48]:


df1=pd.read_csv(r"C:\Users\mahima neema\Desktop\Project 4\data.csv")


# In[49]:


df1.head()


# In[50]:


df1.shape


# Just wanted to see the Target I will be dealing with

# In[51]:


fig, ax = pyplot.subplots(figsize=(20, 20))
sns.countplot(df1['loan_status'], color='red')


# Now wanted to explore my arsenal  for my Target

# In[52]:


plt.figure(figsize = (50,50))
matrix = df1.corr().round(1)
sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag',linewidths=.5)
plt.show()


# None or Almost none Negative Correlations
# 
# Some Positive Correlations
# 
# Well, still don't know if I could shoot the Traget yet!

# In[53]:


df1.info()


# Well, Looks like needs to sharpen some of my weapons to aim better!

# In[54]:


#Dropping the columns that has 90% values missing

nulls = pd.DataFrame(round(df1.isnull().sum()/len(df1.index)*100,2),columns=['null_percent'])
drop_cols = nulls[nulls['null_percent']>90.0].index
df1.drop(drop_cols, axis=1, inplace=True)


# In[55]:


df1.shape


# In[56]:


df1.info()


# In[57]:


df1.head()


# In[58]:


#Checking the null percent of each columns left
nulls = pd.DataFrame(round(df1.isnull().sum()/len(df1.index)*100,2),columns=['null_percent'])

nulls[nulls['null_percent']!=0.00].sort_values('null_percent',ascending=False)


# In[59]:


updated_df = df1


# In[60]:


nulls = pd.DataFrame(round(updated_df.isnull().sum()/len(updated_df.index)*100,2),columns=['null_percent'])

nulls[nulls['null_percent']!=0.00].sort_values('null_percent',ascending=False)


# In[61]:


#upadted all the missing values with 0

updated_df['mths_since_rcnt_il']=updated_df['mths_since_rcnt_il'].fillna(0)
updated_df['open_rv_12m']=updated_df['open_rv_12m'].fillna(0)

updated_df['inq_last_12m']=updated_df['inq_last_12m'].fillna(0)

updated_df['open_rv_24m']=updated_df['open_rv_24m'].fillna(0)
updated_df['total_cu_tl']=updated_df['total_cu_tl'].fillna(0)

updated_df['all_util']=updated_df['all_util'].fillna(0)
updated_df['max_bal_bc']=updated_df['max_bal_bc'].fillna(0)
updated_df['total_bal_il']=updated_df['total_bal_il'].fillna(0)

updated_df['total_bal_il']=updated_df['total_bal_il'].fillna(0)
updated_df['open_il_24m']=updated_df['open_il_24m'].fillna(0)
updated_df['open_il_12m']=updated_df['open_il_12m'].fillna(0)

updated_df['open_act_il']=updated_df['open_act_il'].fillna(0)
updated_df['open_acc_6m']=updated_df['open_acc_6m'].fillna(0)

updated_df['inq_fi']=updated_df['inq_fi'].fillna(0)
updated_df['mths_since_last_record']=updated_df['mths_since_last_record'].fillna(0)
updated_df['mths_since_recent_bc_dlq']=updated_df['mths_since_recent_bc_dlq'].fillna(0)
updated_df['mths_since_last_major_derog']=updated_df['mths_since_last_major_derog'].fillna(0)

updated_df['mths_since_recent_revol_delinq']=updated_df['mths_since_recent_revol_delinq'].fillna(0)
updated_df['mths_since_last_delinq']=updated_df['mths_since_last_delinq'].fillna(0)
updated_df['mths_since_recent_inq']=updated_df['mths_since_recent_inq'].fillna(0)



# In[62]:


nulls = pd.DataFrame(round(updated_df.isnull().sum()/len(updated_df.index)*100,2),columns=['null_percent'])

nulls[nulls['null_percent']!=0.00].sort_values('null_percent',ascending=False)


# In[65]:


#dropping some unnecessary columns like Next payment Date, Zip Code etc which does not add to the predicting power and have null values
updated_df = updated_df.drop(['next_pymnt_d','zip_code','last_pymnt_d','last_credit_pull_d','earliest_cr_line'], axis=1)


# In[66]:


nulls = pd.DataFrame(round(updated_df.isnull().sum()/len(updated_df.index)*100,2),columns=['null_percent'])

nulls[nulls['null_percent']!=0.00].sort_values('null_percent',ascending=False)


# In[67]:


updated_df.shape


# In[68]:


updated_df.info()


# In[69]:


#Still some null values, hence deleting some rows with null values
df = updated_df.dropna(axis=0)


# In[70]:


df.shape


# In[71]:


df.info()


# In[72]:


nulls = pd.DataFrame(round(df.isnull().sum()/len(df.index)*100,2),columns=['null_percent'])

nulls[nulls['null_percent']!=0.00].sort_values('null_percent',ascending=False)


# Now my weapons are Ready but Target is quite a 'Model' and so needs them to be better looking so gotta sharpen them further

# In[73]:


df['term'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
df['term'] = df['term'].astype(int)
set(df["term"])


# In[74]:


df['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
df['emp_length'] = df['emp_length'].astype(int)
set(df["emp_length"])


# In[75]:


credit_default = [1 if i=='Default' or i=='Charged Off' or i=='Late (31-120 days)' else 0 for i in df['loan_status']]
df['credit_default'] = credit_default
df['credit_default'].value_counts()


# Well, My Target seems to be Ready here

# In[78]:


Y = df['credit_default']


# In[79]:


df.info()


# Well some of my weapons are little out of the range for my Target, gotta check their quality.

# In[81]:


df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# In[86]:


fig, ax = pyplot.subplots(figsize=(13, 7))
sns.countplot(df['loan_status'], palette='inferno')
plt.xlabel("Loan Status", fontsize=15)
plt.ylabel("Count of Each Loans Status Category", fontsize=15)


# In[84]:


fig, ax = pyplot.subplots(figsize=(20, 7))
sns.countplot(df['addr_state'], palette='inferno')
plt.xlabel("States", fontsize=15)


# In[94]:


#Aggrgating the amount of loan given belongs to each Category

status = df.groupby(['loan_status'])[['loan_amnt']].agg('mean')
status


# In[99]:


status=df.groupby(['addr_state'])[['loan_amnt']].agg('mean')
status.plot.bar()


# In[148]:


loan_grades = df.groupby("grade").mean().reset_index()

sns.set(rc={'figure.figsize':(15,6)})
sns.barplot(x='grade', y='loan_amnt', data=loan_grades, palette='inferno')
plt.title("Average Loan Amount - Grade", fontsize=20)
plt.xlabel("Loan Grade", fontsize=15)
plt.ylabel("Average Loan Amount", fontsize=15);


# So I explored all the possible Weapons, I figured they could be useful in exploring the range of the Target but won't be useful enough to shoot my Target so Dropping them

# In[100]:


new_df = df
new_df=new_df.drop(['emp_title','addr_state','sub_grade'], axis=1)


# Now for the final Touch, preparing the Final Arsenal will be using for my Target

# In[101]:


from sklearn import preprocessing

new_df = pd.DataFrame(new_df)

new_df= new_df.apply(preprocessing.LabelEncoder().fit_transform)


# In[102]:


new_df.shape


# In[103]:


new_df.info()


# In[104]:


plt.figure(figsize = (50,50))
matrix = new_df.corr().round(1)
sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag',linewidths=.5)
plt.show()


# In[105]:


X = new_df
X=X.drop(['id','credit_default','loan_status'], axis=1)


# In[107]:


plt.figure(figsize = (50,50))
matrix = X.corr().round(1)
sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag',linewidths=.5)
plt.show()


# In[112]:


X.shape


# In[113]:


Y.shape


# Shooting the Traget now 

# Model 1

# In[116]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=5)
clf = clf.fit(X, Y)


# In[117]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, Y, cv=10, scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))


# Uu Oh, Overfitting Alert!!!

# In[118]:


#Checking the important features

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)

indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

names = X.columns

for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, names[indices[f]], importances[indices[f]]))


# In[119]:


'''
Now collection recovery fees is there because there is recovery,now if recovery is there then it is charged offed loan, 
its kinda of repetitive information so removing it'''

X.drop(["recoveries","collection_recovery_fee"], axis=1, inplace=True)


# In[120]:


# Now Trying again

clf = RandomForestClassifier(max_depth=3)
clf = clf.fit(X, Y)
scores = cross_val_score(clf, X, Y, cv=10, scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))


# Little bit feasible score there, but still needs to make sure for over fitting

# In[121]:


importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

names = X.columns

for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, names[indices[f]], importances[indices[f]]))


# In[ ]:


Features Looks Feasible but still needs to check with other models


# In[ ]:





# MODEL 2

# In[122]:


get_ipython().system('pip install lightgbm')
 
# Importing Required Library
import pandas as pd
import lightgbm as lgb
 
# Similarly LGBMRegressor can also be imported for a regression model.
from lightgbm import LGBMClassifier


# In[123]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=42)


# In[124]:


model = lgb.LGBMClassifier(learning_rate=0.90,max_depth=-5,random_state=42)
model.fit(x_train,y_train,eval_set=[(x_test,y_test),(x_train,y_train)],
          verbose=20,eval_metric='logloss')


# In[125]:


y_pred = model.predict(x_test)


# In[126]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))


# In[127]:


y_pred_train = model.predict(x_train)


# Check for Overfitting

# In[342]:


print('Training set score: {:.4f}'.format(model.score(x_train, y_train)))

print('Test set score: {:.4f}'.format(model.score(x_test, y_test)))


# Since both of the accuracy comes in close range, its not overfitting

# Lets chek with Confusion Matrix

# In[128]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])


# Model 3

# Just In case. another model won't harm 

# In[129]:


get_ipython().system('pip3 install xgboost')


# In[130]:


import xgboost as xgb

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.40,random_state=0)

model3 = xgb.XGBClassifier()
model3.fit(X_train, Y_train)
 
# Predicting the Test set results
Y_pred = model3.predict(X_test)
 


# In[131]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(Y_pred, Y_test)
print('XGBoost Model accuracy score: {0:0.4f}'.format(accuracy_score(Y_test, Y_pred)))


# In[132]:


print('Training set score: {:.4f}'.format(model3.score(X_train, Y_train)))

print('Test set score: {:.4f}'.format(model3.score(X_test, Y_test)))


# In[133]:


# Making the Confusion Matrix
CM = confusion_matrix(Y_test, Y_pred)


# In[134]:


print('Confusion matrix\n\n', CM)

print('\nTrue Positives(TP) = ', CM[0,0])
print('\nTrue Negatives(TN) = ', CM[1,1])
print('\nFalse Positives(FP) = ', CM[0,1])
print('\nFalse Negatives(FN) = ', CM[1,0])


# all the Models are performing Good,with the current Data. Now only Fresh Data can tell if the Models are actually Good enough or not!!

# In[ ]:





# In[ ]:




