# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: HEMANTH KUMAR B
RegisterNumber:  21222040047
*/
import pandas as pd
data = pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![image](https://user-images.githubusercontent.com/116530537/204098749-10859744-83f6-4ffc-ad76-f3cee62f9da3.png)

![image](https://user-images.githubusercontent.com/116530537/204098780-b039d6ac-7be4-4edd-a5b0-70889041d107.png)

![image](https://user-images.githubusercontent.com/116530537/204098801-e45b6573-2bdc-4662-868a-ae7530301caa.png)

![image](https://user-images.githubusercontent.com/116530537/204098819-33f225f9-7234-4f2f-ac31-e960c04d2bd5.png)

![image](https://user-images.githubusercontent.com/116530537/204098848-f355c2a2-eb30-4fff-987b-b3fb9816dcca.png)

![image](https://user-images.githubusercontent.com/116530537/204098867-e74cdab2-3e58-4b78-8f91-55da281cfae6.png)

![image](https://user-images.githubusercontent.com/116530537/204098875-2080c7d3-8617-47af-8934-4ce572f344ff.png)

![image](https://user-images.githubusercontent.com/116530537/204098886-4bc02eff-e397-486e-8222-c2847673efbd.png)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
