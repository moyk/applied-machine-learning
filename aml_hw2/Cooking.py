
# coding: utf-8

# In[2]:

import json
from sklearn.naive_bayes import GaussianNB
import nltk
from sklearn import cross_validation

with open('train.json') as data_file:
    data = json.load(data_file)
    CuisineDict = {}
    IngredientsList = []
    res=[]
    labelList=[]
    for each in data:
        temp = [0 for i in range(6714)]
        labelList.append(each['cuisine'])
        if each['cuisine'] not in CuisineDict:
            CuisineDict[each['cuisine']] = 1
        for j in each['ingredients']:
            if j not in IngredientsList:
                IngredientsList.append(j)
            ind=IngredientsList.index(j)
            temp[ind]=1
        res.append(temp)
#     print res[0], len(res)

#     print len(CuisineDict.keys())
#     print len(IngredientsList)


# In[4]:

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
score=cross_val_score(gnb,res,labelList,n_jobs=-1)
print score.mean()


# In[5]:

from sklearn.naive_bayes import BernoulliNB

bb = BernoulliNB()
score=cross_val_score(bb,res,labelList)
print score.mean()


# In[4]:

from sklearn.linear_model import LogisticRegression

score = cross_val_score(LogisticRegression(), res, labelList, cv=3)
print score.mean()


# In[3]:

with open('test.json') as data_file:
    data = json.load(data_file)
    IngredientsList
    test_data=[]
    idList = []
    for each in data:
        temp = [0 for i in range(len(IngredientsList))]
        idList.append(each['id'])
        for j in each['ingredients']:
            if j in IngredientsList:
                ind=IngredientsList.index(j)
                temp[ind]=1
        test_data.append(temp)
    print len(test_data)


# In[13]:

LogReg = LogisticRegression()
LogReg.fit(res, labelList)
test_result = LogReg.predict(test_data)


# In[15]:

print test_result[0]


# In[19]:

import pandas as pd
df = pd.DataFrame({"id" : idList, "cuisine" : test_result})
df.to_csv("cookingResult.csv", index=False)





