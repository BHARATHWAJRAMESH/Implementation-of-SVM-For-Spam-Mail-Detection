# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect a labeled dataset of emails, distinguishing between spam and non-spam.

2.Preprocess the email data by removing unnecessary characters, converting to lowercase, removing stop words, and performing stemming or lemmatization.

3.Extract features from the preprocessed text using techniques like Bag-of-Words or TF-IDF.

4.Split the dataset into a training set and a test set.

5.Train an SVM model using the training set, selecting the appropriate kernel function and hyperparameters.

6.Evaluate the trained model using the test set, considering metrics such as accuracy, precision, recall, and F1 score.

7.Optimize the model's performance by tuning its hyperparameters through techniques like grid search or random search.

8.Deploy the trained and fine-tuned model for real-world use, integrating it into an email server or application.

9.Monitor the model's performance and periodically update it with new data or adjust hyperparameters as needed.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: BHARATHWAJ R
RegisterNumber: 212222240019
*/

import chardet
file = "spam.csv"
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()

data.info()

data.isnull().sum()

x = data["v1"].values
y = data["v2"].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy
```

## Output:

RESULT OUTPUT:

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119394248/290b9792-73e0-42f4-b3a9-8d6ff69f2b27)


data.head()

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119394248/767eb671-ceed-4b26-8661-ed25bdfecb5c)



data.info()

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119394248/5e951e12-f30a-45b5-ae92-4d0f03204aa2)



data.isnull().sum()

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119394248/f875166c-fff0-4271-976f-54ab4e16f079)



Y_prediction value

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119394248/a8a47773-9696-4dcb-9d24-0d02c54e3674)



Accuracy value

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119394248/adaa4ce5-a898-43f7-a388-91e39e9ca02a)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
