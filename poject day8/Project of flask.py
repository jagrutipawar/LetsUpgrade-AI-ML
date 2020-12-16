#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from warnings import filterwarnings
filterwarnings("ignore")


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# In[2]:


data=pd.read_csv("diabetes.csv")
data.head()


# # __EDA__

# In[3]:


data.info()


# In[4]:


data.shape


# # five point summary

# In[5]:


data.describe()


# In[6]:


# Checking mising values
data.isna().sum().sum()


# In[7]:


data.hist()


# In[8]:


fig , axe = plt.subplots(nrows=5,ncols=2,figsize=(10,15))
axe = axe.flatten()
sns.distplot(data['Pregnancies'],ax=axe[0])
sns.distplot(data['Glucose'],ax=axe[1])
sns.distplot(data['BloodPressure'],ax=axe[2])
sns.distplot(data['SkinThickness'],ax=axe[3])
sns.distplot(data['Insulin'],ax=axe[4])
sns.distplot(data['BMI'],ax=axe[5])
sns.distplot(data['DiabetesPedigreeFunction'],ax=axe[6])
sns.distplot(data['Age'],ax=axe[7])
sns.countplot(data['Outcome'],ax=axe[8])
fig.tight_layout()
fig.show()
axe.flat[-1].set_visible(False)


# In[9]:


fig , axe = plt.subplots(nrows=4,ncols=2,figsize=(10,15))
axe = axe.flatten()
sns.boxplot(data['Pregnancies'],ax=axe[0])
sns.boxplot(data['Glucose'],ax=axe[1])
sns.boxplot(data['BloodPressure'],ax=axe[2])
sns.boxplot(data['SkinThickness'],ax=axe[3])
sns.boxplot(data['Insulin'],ax=axe[4])
sns.boxplot(data['BMI'],ax=axe[5])
sns.boxplot(data['DiabetesPedigreeFunction'],ax=axe[6])
sns.boxplot(data['Age'],ax=axe[7])
fig.tight_layout()
fig.show()


# # Handling Outlier after mean replacement

# In[10]:


data1 = data.copy()
mean_insulin = float(data1['Insulin'].mean())
data1['Insulin'] = np.where(data1['Insulin'] > np.percentile(data1['Insulin'],85),mean_insulin,data1['Insulin'])


# In[11]:


mean_dpf = float(data1['DiabetesPedigreeFunction'].mean())
data1['DiabetesPedigreeFunction'] = np.where(data1['DiabetesPedigreeFunction'] > np.percentile(data1['DiabetesPedigreeFunction'],85),
                                             mean_dpf,data1['DiabetesPedigreeFunction'])


# In[12]:


#Checking outlier after mean replacement

fig , axe = plt.subplots(nrows=4,ncols=2,figsize=(10,15))
axe = axe.flatten()
sns.boxplot(data1['Pregnancies'],ax=axe[0])
sns.boxplot(data1['Glucose'],ax=axe[1])
sns.boxplot(data1['BloodPressure'],ax=axe[2])
sns.boxplot(data1['SkinThickness'],ax=axe[3])
sns.boxplot(data1['Insulin'],ax=axe[4])
sns.boxplot(data1['BMI'],ax=axe[5])
sns.boxplot(data1['DiabetesPedigreeFunction'],ax=axe[6])
sns.boxplot(data1['Age'],ax=axe[7])
fig.tight_layout()
fig.show()


# # corellation of all features

# In[13]:


plt.figure(figsize=(12,10))
sns.heatmap(data1.corr(),annot=True,vmax=1,vmin=-1)


# In[14]:


sns.pairplot(data1)


# In[15]:


X = data1.drop('Outcome',axis=1)
y = data1['Outcome']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.40,random_state=1)


# # Logistic Regration model

# In[16]:


logic_r = LogisticRegression(solver='liblinear')
logic_r.fit(X_train,y_train)
y_pred = logic_r.predict(X_test)
LR_accuracy = accuracy_score(y_test,y_pred)
print('\033[1m''*'*63)
print('\033[1m''Confusion Matrix\n',confusion_matrix(y_test,y_pred))
print('-'*40)
print('Accuracy of Logistic Regression :{:.2f}'.format(LR_accuracy))
print('-'*40)
print('\n Classification Report\n',classification_report(y_test,y_pred))
print('*'*63)


# # Naive Bayes Model

# In[17]:


NB = GaussianNB()
NB.fit(X_train,y_train)
y_pred = NB.predict(X_test)
NB_accuracy = accuracy_score(y_test,y_pred)
print('\033[1m''*'*63)
print('\033[1m''Confusion Matrix\n',confusion_matrix(y_test,y_pred))
print('-'*30)
print('Accuracy of Naive Bayes :{:.2f}'.format(NB_accuracy))
print('-'*30)
print('\n Classification Report\n',classification_report(y_test,y_pred))
print('*'*63)


# # KNN Model

# In[18]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
KNN_accuracy = accuracy_score(y_test,y_pred)
print('\033[1m''*'*63)
print('\033[1m''Confusion Matrix\n',confusion_matrix(y_test,y_pred))
print('-'*30)
print('Accuracy of KNN :{:.2f}'.format(KNN_accuracy))
print('-'*30)
print('\n Classification Report\n',classification_report(y_test,y_pred))
print('*'*63)


# # SVC model

# In[19]:


svc = SVC()
svc.fit(X_train,y_train)
predicted_svc = svc.predict(X_test)
SVC_accuracy = accuracy_score(y_test,predicted_svc)
print('\033[1m''*'*63)
print('\033[1m''Confusion Matrix\n',confusion_matrix(y_test,predicted_svc))
print('-'*30)
print('Accuracy of SVC :',SVC_accuracy)
print('-'*30)
print('\n Classification Report\n',classification_report(y_test,predicted_svc))
print('*'*63)


# # Decision Tree Model

# In[20]:


dTree = DecisionTreeClassifier(criterion = 'gini', random_state=1)
dTree.fit(X_train, y_train)
predicted_DT = dTree.predict(X_test)
DT_accuracy = accuracy_score(y_test,predicted_DT)
print('\033[1m''*'*63)
print('\033[1m''Confusion Matrix\n',confusion_matrix(y_test,predicted_DT))
print('-'*30)
print('Accuracy of Decision Tree :{:.2f}'.format(DT_accuracy))
print('-'*30)
print('\n Classification Report\n',classification_report(y_test,predicted_DT))
print('*'*63)


# # Pruned decision Tree

# In[21]:


dTreeR = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, random_state=1)
dTreeR.fit(X_train, y_train)
predicted_DTR = dTreeR.predict(X_test)
DTR_accuracy = accuracy_score(y_test,predicted_DTR)
print('\033[1m''*'*63)
print('\033[1m''Confusion Matrix\n',confusion_matrix(y_test,predicted_DTR))
print('-'*30)
print('Accuracy of Decision Tree with Regularization:{:.2f}'.format(DTR_accuracy))
print('-'*30)
print('\n Classification Report\n',classification_report(y_test,predicted_DTR))
print('*'*63)


# # Bagging Model

# In[22]:


dTreeR = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, random_state=1)
dTreeR.fit(X_train, y_train)
predicted_DTR = dTreeR.predict(X_test)
DTR_accuracy = accuracy_score(y_test,predicted_DTR)
print('\033[1m''*'*63)
print('\033[1m''Confusion Matrix\n',confusion_matrix(y_test,predicted_DTR))
print('-'*30)
print('Accuracy of Decision Tree with Regularization:{:.2f}'.format(DTR_accuracy))
print('-'*30)
print('\n Classification Report\n',classification_report(y_test,predicted_DTR))
print('*'*63)


# # Adapting Boosting Model

# In[23]:


adab = AdaBoostClassifier(n_estimators=50, random_state=1)
adab = adab.fit(X_train, y_train)
predicted_ADA = adab.predict(X_test)
ADA_accuracy = accuracy_score(y_test,predicted_ADA)
print('\033[1m''*'*63)
print('\033[1m''Confusion Matrix\n',confusion_matrix(y_test,predicted_ADA))
print('-'*30)
print('Accuracy of AdaBoostClassifier :{:.2f}'.format(ADA_accuracy))
print('-'*30)
print('\n Classification Report\n',classification_report(y_test,predicted_ADA))


# # Gradient Boosting Model

# In[24]:


gradb = GradientBoostingClassifier(n_estimators = 100,random_state=1)
gradb = gradb.fit(X_train, y_train)
predicted_GRAD = gradb.predict(X_test)
GRAD_accuracy = accuracy_score(y_test,predicted_GRAD)
print('\033[1m''*'*63)
print('\033[1m''Confusion Matrix\n',confusion_matrix(y_test,predicted_GRAD))
print('-'*30)
print('Accuracy of GradientBoostingClassifier :{:.2f}'.format(GRAD_accuracy))
print('-'*30)
print('\n Classification Report\n',classification_report(y_test,predicted_GRAD))
print('*'*63)


# # Random Forest Model

# In[ ]:


randf = RandomForestClassifier(n_estimators = 100, random_state=1, max_features=3)
randf = randf.fit(X_train, y_train)
predicted_RAN = randf.predict(X_test)
RAN_accuracy = accuracy_score(y_test,predicted_RAN )
print('\033[1m''*'*63)
print('\033[1m''Confusion Matrix\n',confusion_matrix(y_test,predicted_RAN ))
print('-'*30)
print('Accuracy of RAN :{:.2f}'.format(RAN_accuracy))
print('-'*30)
print('\n Classification Report\n',classification_report(y_test,predicted_RAN ))
print('*'*63)


# # Model Performance

# In[ ]:


Scores = [('Naive bayes', NB_accuracy),
      ('KNN', KNN_accuracy),
      ('Logistic Regression', LR_accuracy),
      ('SVC', SVC_accuracy ),
      ('Decision Tree',DT_accuracy),
      ('Decision Tree with Regularization',DTR_accuracy),
      #('Bagging',BAG_accuracy),
      ('Adaptive Boosting',ADA_accuracy),
      ('Gradient Boosting',GRAD_accuracy),
      ('Random Forest',RAN_accuracy)]
Scores = pd.DataFrame(Scores,columns=['Model','Accuracy score'])
Scores.sort_values(by='Accuracy score',ascending=False)


# # Saving The Model

# In[ ]:


pickle.dump(adab,open('model.pkl','wb'))


# In[ ]:




