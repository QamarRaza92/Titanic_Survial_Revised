import pandas as pd
import numpy as np

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
train = pd.read_csv("C:\\Users\\moham\\Downloads\\titanic.csv")
print("shape of train : ",train.shape)
train.sample()


df = train.drop(columns = ['Name','PassengerId','Ticket'],axis=1)
print(df.sample())


df['Family Members'] = df['SibSp'] + df['Parch']
df.drop(columns = ['SibSp','Parch'],axis=1,inplace=True)
print(df.sample())


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
sns.kdeplot(df['Age'])
plt.show()
sns.histplot(df['Embarked'])
plt.show()


df['Embarked'].fillna('Unknown',inplace=True)
print(df.isnull().sum())


df = df[df['Embarked'] != 'Unknown']
print(df.shape)
print(df.isnull().sum())


df.drop(columns=['Cabin'],axis=1,inplace=True)
print(df.shape)


print(df.shape)
print(df.isnull().sum())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['Survived'],axis=1), df['Survived'], random_state=0, test_size=0.2)


print(f"X_train Shape : {X_train.shape}\nX_test Shape : {X_test.shape}\ny_train Shape : {y_train.shape}\ny_test Shape : {y_test.shape}")


print(X_train.sample())
print(X_test.sample())


trf1 = ColumnTransformer([
                            ('OneHotEncoding',OneHotEncoder(drop='first',sparse_output=False),['Sex','Embarked']),
                            ('Imputing',SimpleImputer(strategy='median'),['Age']),
                            ('MinMaxScaling',MinMaxScaler(),['Fare'])
                          ]
                          ,remainder='passthrough')


from sklearn.ensemble import RandomForestClassifier
pipe = Pipeline(steps=[('transforming',trf1),('RandomForestClassification',RandomForestClassifier())])
pipe.fit(X_train,y_train)


ypred = pipe.predict(X_test)
print('the accuracy score is : ',accuracy_score(ypred,y_test)*100,'%')


import pickle 
with open('titanic.pkl','wb') as file:
    pickle.dump(pipe,file)

