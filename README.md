# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Use the standard libraries in python for Gradient Design.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing the graph.
5. Predict the regression for marks by using the representation of the graph.
6. compare the graphs and hence we obtained the linear regression for the given data.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Andra likitha
RegisterNumber:  212221220006
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
#displaying the content in datafile
print("df.head():")
df.head()

print("df.tail():")
df.tail()

#Segregating data to variables
print("Array value of X:")
X=df.iloc[:,:-1].values
X

print("Array value of X:")
Y=df.iloc[:,1].values
Y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
print("Values of Y prediction:")
Y_pred

#displaying actual values
print("Array values of Y test:")
Y_test

#graph plot for training data
print("Training set graph:")
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
print("Test set graph:")
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

print("Values of MSE,MAE and RMSE:")
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
print('Values of MSE'
```


## Output:
df.head():
![229293862-34870b9a-d46d-41cb-9ea0-13cea22f00ae](https://user-images.githubusercontent.com/131592130/234187696-bfec141b-5e55-4a60-a257-7c46191cf70d.png)

df.tail():
![229293967-09c262a2-1793-4fa3-9ad4-5b969c669900](https://user-images.githubusercontent.com/131592130/234187696-bfec141b-5e55-4a60-a257-7c46191cf70d.png)

Array value of X:
![229293469-8f08eb90-7774-46aa-b0db-0ac8df78dad3](https://user-images.githubusercontent.com/131592130/234188032-8fb2a023-7e7b-4379-a114-7006b9ca6bc3.png)

Array value of Y:
![229293494-aa427d62-0d42-4747-9b9d-474c5f58fb29](https://user-images.githubusercontent.com/131592130/234188438-205a8de2-f326-4312-a074-677c4d1f7f51.png)

Values of Y prediction:
![229293514-ef09d849-1b86-4783-b366-9552fbafecca](https://user-images.githubusercontent.com/131592130/234188563-167eef20-0913-4804-a62e-8c073c4d89b8.png)

Array values of Y test:
![229293545-32b41b3c-8494-4138-8f49-6161ed6af60b](https://user-images.githubusercontent.com/131592130/234188695-7f79dc93-c59b-4ee4-ac01-d6e810bc2da2.png)

Training set graph:
![229293565-fbd372b3-aac6-4ed6-be0b-87905f046ebb](https://user-images.githubusercontent.com/131592130/234188836-d8fb02b9-e703-472a-9243-858b4b8267df.png)

Test set graph:
![229293588-9a4d9a34-3f38-4e5f-bd77-3e3f79bb2cf0](https://user-images.githubusercontent.com/131592130/234188980-35889b6b-6863-4f7d-9f34-caaa78207c98.png)

Values of MSE,MAE and RMSE:
![229293605-f9e791d8-b7c0-45c1-ac1b-2901364e8b26](https://user-images.githubusercontent.com/131592130/234189112-e7640950-0eb9-4569-b837-3097a6d444f0.png)






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
