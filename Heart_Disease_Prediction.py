#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


data=pd.read_csv('C:\\Users\\DIBYAKANTI\\Downloads\\heart attack prediction.csv')
data
data.info()


# In[3]:


df=pd.DataFrame(data)
print(df)
df1=df.dropna()
print(df1)


# In[4]:


import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns


# In[5]:


pd.crosstab(df1.age,df1.TenYearCHD).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[6]:


age_unique=sorted(df1.age.unique())
age_heartRate_values=df1.groupby('age')['heartRate'].count().values
mean_heartRate=[]
for i,age in enumerate(age_unique):
    mean_heartRate.append(sum(df1[df1['age']==age].heartRate)/age_heartRate_values[i])
    
plt.figure(figsize=(10,5))
sns.pointplot(x=age_unique,y=mean_heartRate,color='red',alpha=0.8)
plt.xlabel('Age',fontsize = 15,color='blue')
plt.xticks(rotation=45)
plt.ylabel('Heart Rate',fontsize = 15,color='blue')
plt.title('Age vs heartRate',fontsize = 20,color='black')
plt.grid()
plt.show()


# In[7]:


#Parameter to be choosen

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [2,4]
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the param grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[8]:


from sklearn.model_selection import train_test_split
y=df1["TenYearCHD"]
X=df1.drop('TenYearCHD',axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[9]:


from sklearn.model_selection import GridSearchCV
rf_Model=RandomForestClassifier()
rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 3, verbose=0, n_jobs = 4)
#Fitting the model on the training set
rf_Grid.fit(X_train,y_train)


# In[10]:


#Slecting the best parameter and the predicting our target

print(rf_Grid.best_estimator_)
print('\n')
print('-'*70)
print('\n')
print(rf_Grid.best_params_)

#Let's advance to the prediction phase

rf_Grid.best_estimator_.fit(X_train,y_train)
y_pred=rf_Grid.best_estimator_.predict(X_test)
print('\n')
print('-'*70)
print('\n')
print(f'{accuracy_score(y_test,y_pred)*100}')


# In[11]:


#loading dataset
import pandas as pd
import numpy as np
#visualisation

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('whitegrid')
# data preprocessing
from sklearn.preprocessing import StandardScaler
# data splitting
from sklearn.model_selection import train_test_split
# data modeling
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#ensembling
from mlxtend.classifier import StackingCVClassifier


# In[12]:


class classifieur:
    
    """In order to combine to flexibility and speed execution of Scikit learn we implement a class of method whre
     the objective will be to call all the previous step into one unique cell"""
    
    def __init__(self,estimators):
        self.algo=estimators
        self.best_estimateurs={}
    #Fitting the models before the prediction phase
    
    def fit(self,X_train,y_train):
        for estim in self.algo:
            self.algo[estim]['mod'].fit(X_train,y_train)
            pred=self.algo[estim]['mod'].score(X_train,y_train)*100
            self.best_estimateurs[estim]={"score":pred}
    
#     Showing the score related to the training part in order to verify if there's any underfit case 
    def score_train(self):
        for estim in self.algo:
            print(f'------{estim}--------')
            print(f"Le score obtenu pour {estim} sur le train set est : {self.best_estimateurs[estim]['score']}")
    
    
    #Prediction used on the test set for all our models
    def pred(self,X_train,y_train,X_test,y_test):
        pred={}
        for estim in self.algo:
            self.algo[estim]['mod'].fit(X_train,y_train)
            pred[estim]=self.algo[estim]['mod'].predict(X_test)
        return(pred)
    
    #Score lié à la qualité de nos prédiction
    def show_score(self,y_test,pred):
        conf_mat={}
        acc_score={}
        for estim in self.algo:
            print(f"--------------{estim}-------------------")
            conf_mat[estim]=confusion_matrix(y_test,pred[estim])
            acc_score[estim]=accuracy_score(y_test,pred[estim])
            print(f'En faisant appel à la matrice de confusion on a : \n {conf_mat[estim]}')
            print('\n')
            print(f'Le score de précision de {estim} est de :\n {acc_score[estim]*100}')
            print('\n')
            print('On présente le rapport de classification avec toutes les métriques liées au score prédictif')
            print(classification_report(y_test,pred[estim]))
            print(f"*"*40)
            print("\n")
        return(acc_score)
    
    def roc_curve(self,pred):
        plt.figure(figsize=(10,5))
        for estim in pred.keys():
            fpr,tpr,threshold = roc_curve(y_test,pred[estim])
            sns.set_style('whitegrid')
            plt.title('Reciver Operating Characterstic Curve')
            plt.plot(fpr,tpr,label=f'{estim}')
        plt.ylabel('True positive rate')
        plt.xlabel('False positive rate')
        plt.legend()
        plt.show()
        
    def model_eval(self,acc_score):
        accuracy=acc_score.copy()
        accuracy.update((x,y*100) for x,y in acc_score.items())
        models=list(acc_score.keys())
        scores=list(accuracy.values())
        mod_ev=pd.DataFrame({'Model':models,'Accuracy':scores})
        mod_ev
        return(mod_ev)

# Estimators to be called 
estimators={"knn":{'mod':KNeighborsClassifier(),"param":0},
                 "logistic regression":{'mod':LogisticRegression(),'param':0},
           "random forest":{'mod':RandomForestClassifier(max_depth=4, min_samples_split=5, n_estimators=33),"param":0},
           "decision trees":{"mod":DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth = 6),"param":0},
                 'naive bayes':{"mod":GaussianNB(),"param":0}}


# In[13]:


classification=classifieur(estimators)
pred=classification.pred(X_train,y_train,X_test,y_test)
acc_score=classification.show_score(y_test,pred)


# In[14]:


classification.roc_curve(pred)


# In[15]:


model_ev=classification.model_eval(acc_score)
model_ev


# In[29]:


colors = ['red','green','blue','gold','silver','yellow','orange','skyblue']
plt.figure(figsize=(12,5))
plt.title("Accuracy of different models",fontsize=20)
plt.xlabel("Accuracy %",fontsize=20)
plt.ylabel("Algorithms",fontsize=20)
plt.bar(model_ev['Model'],model_ev['Accuracy'],color = colors)
plt.show()


# In[18]:


# Initialisons d'abords nos modèles
knn=estimators['knn']['mod']


# In[ ]:


scv=StackingCVClassifier(classifiers=[xgb,knn,svc],meta_classifier= knn,random_state=50)
scv.fit(X_train,y_train)
scv_predicted = scv.predict(X_test)
scv_conf_matrix = confusion_matrix(y_test, scv_predicted)
scv_acc_score = accuracy_score(y_test, scv_predicted)
print("confussion matrix")
print(scv_conf_matrix)
print("\n")
print("Accuracy of StackingCVClassifier:",scv_acc_score*100,'\n')
print(classification_report(y_test,scv_predicted))


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
from keras import regularizers


# In[ ]:


from keras import backend as K
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(15, input_dim=15, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(7, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid')) 
    return model
model = create_model()
adam = Adam(lr=0.00001)
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
print(model.summary())


# In[ ]:


history=model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=30, batch_size = 20, verbose=2)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy',fontsize=15)
plt.ylabel('accuracy',fontsize=15)
plt.xlabel('epoch',fontsize=15)
plt.legend(['train', 'test'])
plt.show()


# In[ ]:


# Model Losss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss',fontsize=15)
plt.ylabel('loss',fontsize=15)
plt.xlabel('epoch',fontsize=15)
plt.legend(['train', 'test'])
plt.show()


# In[ ]:


from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
y_pred_bool = np.argmax(y_pred, axis=1)
print(confusion_matrix(y_test, y_pred_bool))
print(classification_report(y_test, y_pred_bool))


# In[ ]:


from collections import Counter
print(y_test.unique())
Counter(y_train)


# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report
from sklearn.linear_model import LogisticRegression
m1='Logistic Regression'
lr=LogisticRegression()
model1=lr.fit(X_train,y_train)
lr_predicted_prb = lr.predict_proba(X_test)
lr_predict=lr.predict(X_test)
lr_conf_matrix=confusion_matrix(y_test,lr_predict)
lr_acc_score=accuracy_score(y_test,lr_predict)
print("confussion matrix of logistic regression")
print(lr_conf_matrix)
print("\n")
print("Accuracy of Logistic Regression:",lr_acc_score*100,'\n')
print(classification_report(y_test,lr_predict))


# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[ ]:


m2 = 'Naive Bayes'
nb = GaussianNB()
nb.fit(X_train,y_train)
nbpred = nb.predict(X_test)
nb_predicted_prb = nb.predict_proba(X_test)
nb_conf_matrix = confusion_matrix(y_test, nbpred)
nb_acc_score = accuracy_score(y_test, nbpred)
print("confussion matrix")
print(nb_conf_matrix)
print("\n")
print("Accuracy of Naive Bayes model:",nb_acc_score*100,'\n')
print(classification_report(y_test,nbpred))


# In[ ]:


m3 = 'Random Forest Classfier'
rf = RandomForestClassifier(n_estimators=7, random_state=11,max_depth=5)
rf.fit(X_train,y_train)
rf_predicted = rf.predict(X_test)
rf_predicted_prb = rf.predict_proba(X_test)
rf_conf_matrix = confusion_matrix(y_test, rf_predicted)
rf_acc_score = accuracy_score(y_test, rf_predicted)
print("confussion matrix")
print(rf_conf_matrix)
print("\n")
print("Accuracy of Random Forest:",rf_acc_score*100,'\n')
print(classification_report(y_test,rf_predicted))


# In[ ]:


from sklearn import metrics
m5 = 'K-NeighborsClassifier'
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
knn_predicted = knn.predict(X_test)
knn_predicted_prb = knn.predict_proba(X_test)
knn_conf_matrix = confusion_matrix(y_test, knn_predicted)
knn_acc_score = accuracy_score(y_test, knn_predicted)
print("confussion matrix")
print(knn_conf_matrix)
print("\n")
print("Accuracy of DecisionTreeClassifier:",knn_acc_score*100,'\n')
print(classification_report(y_test,knn_predicted))


# In[ ]:


m6 = 'DecisionTreeClassifier'
dt = DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth = 6)
dt.fit(X_train, y_train)
dt_predicted = dt.predict(X_test)
dt_predicted_prb = dt.predict_proba(X_test)
dt_conf_matrix = confusion_matrix(y_test, dt_predicted)
dt_acc_score = accuracy_score(y_test, dt_predicted)
print("confussion matrix")
print(dt_conf_matrix)
print("\n")
print("Accuracy of DecisionTreeClassifier:",dt_acc_score*100,'\n')
print(classification_report(y_test,dt_predicted))


# In[ ]:


import seaborn as sns
from matplotlib import pyplot as plt
lr_false_positive_rate,lr_true_positive_rate,lr_threshold = roc_curve(y_test,lr_predicted_prb[:,1])
nb_false_positive_rate,nb_true_positive_rate,nb_threshold = roc_curve(y_test,nb_predicted_prb[:,1])
rf_false_positive_rate,rf_true_positive_rate,rf_threshold = roc_curve(y_test,rf_predicted_prb[:,1])                                                             
knn_false_positive_rate,knn_true_positive_rate,knn_threshold = roc_curve(y_test,knn_predicted_prb[:,1])
dt_false_positive_rate,dt_true_positive_rate,dt_threshold = roc_curve(y_test,dt_predicted_prb[:,1])
sns.set_style('whitegrid')
plt.figure(figsize=(10,5))
plt.title('Reciver Operating Characterstic Curve', fontsize=15)
plt.plot(lr_false_positive_rate,lr_true_positive_rate,label='Logistic Regression')
plt.plot(nb_false_positive_rate,nb_true_positive_rate,label='Naive Bayes')
plt.plot(rf_false_positive_rate,rf_true_positive_rate,label='Random Forest')
plt.plot(knn_false_positive_rate,knn_true_positive_rate,label='K-Nearest Neighbor')
plt.plot(dt_false_positive_rate,dt_true_positive_rate,label='Desion Tree')
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.ylabel('True positive rate', fontsize=20)
plt.xlabel('False positive rate', fontsize=20)
plt.legend()
plt.show()


# In[ ]:




