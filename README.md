# Implementation of Univariate Linear Regression
## AIM:
To implement univariate Linear Regression to fit a straight line using least squares.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y.
2. Calculate the mean of the X -values and the mean of the Y -values.
3. Find the slope m of the line of best fit using the formula. 
<img width="231" alt="image" src="https://user-images.githubusercontent.com/93026020/192078527-b3b5ee3e-992f-46c4-865b-3b7ce4ac54ad.png">
4. Compute the y -intercept of the line by using the formula:
<img width="148" alt="image" src="https://user-images.githubusercontent.com/93026020/192078545-79d70b90-7e9d-4b85-9f8b-9d7548a4c5a4.png">
5. Use the slope m and the y -intercept to form the equation of the line.
6. Obtain the straight line equation Y=mX+b and plot the scatterplot.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: DHARSHENI K
RegisterNumber: 212224040073
import pandas as pd
 import numpy as np
 data=pd.read_csv("Placement_Data.csv")
 data.head()
 data1=data.copy()
 data1.head()
 data1=data.drop(['sl_no','salary'],axis=1)
 data1
 from sklearn.preprocessing import LabelEncoder
 le=LabelEncoder()
 data1["gender"]=le.fit_transform(data1["gender"])
 data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
 data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
 data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
 data1["degree_t"]=le.fit_transform(data1["degree_t"])
 data1["workex"]=le.fit_transform(data1["workex"])
 data1["specialisation"]=le.fit_transform(data1["specialisation"])
 data1["status"]=le.fit_transform(data1["status"])
 X=data1.iloc[:,: -1]
 Y=data1["status"]
 theta=np.random.randn(X.shape[1])
 y=Y
 def sigmoid(z):
   return 1/(1+np.exp(-z))
 def loss(theta,X,y):
   h=sigmoid(X.dot(theta))
   return -np.sum(y*np.log(h)+ (1-y) * np.log(1-h))
 def gradient_descent(theta,X,y,alpha,num_iterations):
   m=len(y)
   for i in range(num_iterations):
     h=sigmoid(X.dot(theta))
     gradient=X.T.dot(h-y)/m
     theta-=alpha*gradient
   return theta
 theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
 def predict(theta,X):
   h=sigmoid(X.dot(theta))
   y_pred=np.where(h>=0.5 , 1,0)
   return y_pred
 y_pred=predict(theta,X)
 accuracy=np.mean(y_pred.flatten()==y)
 print("Accuracy:",accuracy)
 print("Predicted:\n",y_pred)
 print("Actual:\n",y.values)
 xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
 y_prednew=predict(theta,xnew)
 print("Predicted Result:",y_prednew)
 data.head()
 
*/
```

## Output:
![best fit line](sam.png)


## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
