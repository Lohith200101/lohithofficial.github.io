import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

data=pd.read_csv("salary.csv")
data.head()


x=data.iloc[:, :4]
y=data.iloc[:,-1]


def convert(word):
    word_dict={"btech":0,"mtech":1,"phd":2}
    return word_dict[word]
x["qualification"]=x["qualification"].apply(lambda x:convert(x))

def cast(word):
    word_dict={"lecturer":0,"assistantprofessor":1,"labassistant":2,"professor":3,"hod":4}
    return word_dict[word]
x["position"]=x["position"].apply(lambda x:cast(x))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=41)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

pickle.dump(reg,open("model1.pkl","wb"))
model=pickle.load(open("model1.pkl","rb"))

print(reg.predict([[2,10,4,4]]))

print(reg.score(x_test,y_test))
