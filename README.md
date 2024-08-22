# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import the standard Libraries. 
#### 2.Set variables for assigning dataset values. 
#### 3.Import linear regression from sklearn. 
#### 4.Assign the points for representing in the graph. 
#### 5.Predict the regression for marks by using the representation of the graph. 
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:

### P PARTHIBAN
### 212223230145
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("student_scores.csv")

print(df.tail())
print(df.head())
df.info()

x = df.iloc[:, :-1].values  # Hours
y = df.iloc[:,:-1].values   # Scores

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

print("X_Training:", x_train)
print("X_Test:", x_test)
print("Y_Training:", y_train)
print("Y_Test:", y_test)

reg = LinearRegression()
reg.fit(x_train, y_train)

Y_pred = reg.predict(x_test)

print("Predicted Scores:", Y_pred)
print("Actual Scores:", y_test)

a = Y_pred - y_test
print("Difference (Predicted - Actual):", a)

plt.scatter(x_train, y_train, color="green")
plt.plot(x_train, reg.predict(x_train), color="red")
plt.title('Training set (Hours vs Scores)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test, y_test, color="blue")
plt.plot(x_test, reg.predict(x_test), color="green")
plt.title('Testing set (Hours vs Scores)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mae = mean_absolute_error(y_test, Y_pred)
mse = mean_squared_error(y_test, Y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
```

## Output:
### head():
![image](https://github.com/user-attachments/assets/b81ff787-f053-41e4-bfcb-2db1ec6f6875)

### tail():
![image](https://github.com/user-attachments/assets/21029500-22d8-4848-b8d5-ab5b3e740dca)

## TRAINING SET INPUT
### X_Training:
![image](https://github.com/user-attachments/assets/bc9fe391-f491-4dfb-8874-f3f4235c1c65)

### Y_Training:
![image](https://github.com/user-attachments/assets/348305d5-b7cb-42cb-808e-5a8dd533df5e)

## TEST SET VALUE
### X_Test:
![image](https://github.com/user-attachments/assets/8f0f9121-22b4-4e0d-8f9c-193fe427f647)

### Y_Test:
![image](https://github.com/user-attachments/assets/65ac43d6-870e-4095-81e1-bc1afe7f7cc1)

## TRAINING SET:
![image](https://github.com/user-attachments/assets/bd377bd1-b2f6-4bbd-9d47-9cbd7e0c290c)


### TEST SET:
![image](https://github.com/user-attachments/assets/b1e3213a-2cc7-4e30-add5-9df89fe5c10c)

### MEAN SQUARE ERROR, MEAN ABSOLUTE ERROR AND RMSE:
![image](https://github.com/user-attachments/assets/1ebdbc49-3515-48c3-951f-8c662ebfa198)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
