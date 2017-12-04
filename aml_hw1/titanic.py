import csv
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from collections import defaultdict
from numpy import linalg as LA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def age_approx(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

filepath = "/Users/kellywang/Documents/CornellTech/AppliedMachineLearning/hm1/titanicData/train.csv"
testDataPath = "/Users/kellywang/Documents/CornellTech/AppliedMachineLearning/hm1/titanicData/test.csv"
titanic = pd.read_csv(filepath)

#process test data
testData = pd.read_csv(testDataPath)
testData = testData.drop(['PassengerId','Name','Ticket','Cabin'], 1)
testData['Age'] = testData[['Age', 'Parch']].apply(age_approx, axis=1)
gendertest = pd.get_dummies(testData['Sex'],drop_first=True)
embark_location_test = pd.get_dummies(testData['Embarked'],drop_first=True)
testData.drop(['Sex', 'Embarked'],axis=1,inplace=True)
test_dmy = pd.concat([testData,gendertest,embark_location_test],axis=1)
sb.heatmap(test_dmy.corr())
test_dmy.drop(['Fare', 'Pclass'],axis=1,inplace=True)


#Process training data

titanic.columns = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
sb.countplot(x='Survived',data=titanic, palette='hls')
titanic_data = titanic.drop(['PassengerId','Name','Ticket','Cabin'], 1)
# sb.boxplot(x='Pclass', y='Age', data=titanic_data, palette='hls')
titanic_data['Age'] = titanic_data[['Age', 'Parch']].apply(age_approx, axis=1)
titanic_data.dropna(inplace=True)
gender = pd.get_dummies(titanic_data['Sex'],drop_first=True)
embark_location = pd.get_dummies(titanic_data['Embarked'],drop_first=True)
titanic_data.drop(['Sex', 'Embarked'],axis=1,inplace=True)
# print titanic_data.head()
titanic_dmy = pd.concat([titanic_data,gender,embark_location],axis=1)
sb.heatmap(titanic_dmy.corr())
titanic_dmy.drop(['Fare', 'Pclass'],axis=1,inplace=True)
titanic_dmy.info()

X = titanic_dmy.ix[:,(1,2,3,4,5,6)].values
y = titanic_dmy.ix[:,0].values
X_test = test_dmy.ix[:,:].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)
LogReg = LogisticRegression()
LogReg.fit(X, y)
y_pred = LogReg.predict(X_test)
print y_pred
a = np.array([i for i in range(892,1309+1)])
df = pd.DataFrame({"PassengerId" : a, "Survived" : y_pred})
df.to_csv("result.csv", index=False)
print(classification_report(y_test, y_pred))




