# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:53:32 2023

@author: cwbong
"""

from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# import os
# import numpy as np
# import sklearn

# Loading data (can combine data if necessary)
df = pd.read_csv('./lab7-work/weight-height.csv')
# df.shape
# df.describe()
# df.head()
# df.tail()
# df.nunique()
# df.dtypes

# Data Cleaning:First step of Data Pre-processing
# Remove irrelevant or duplicate data
# df.drop(['col1', 'col2'], axis=1, inplace=True)
# df.drop_duplicates(inplace=True)

# Check for missing data and fix inconsistencies
df.isnull().sum()
# df['Gender'].value_counts()

# Data Reduction: third steps for Data Pre-processing
# Remove less important data
# df.drop(['col5', 'col6'], axis=1, inplace=True)
# Remove duplicates
df.drop_duplicates(keep='first', inplace=True)

print(df.shape)
for col in ['Gender']:
    df[col] = df[col].astype('category')

# df.dtypes
# Data Sampling (last proess of data preprocessing)
X_train, X_test, y_train, y_test = train_test_split(df[["Weight", "Height"]],
                                                    df['Gender'], test_size=0.35, random_state=50)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# y_train.head()

# Data Transformation in Data Pre-processing
# Convert categorical variables to numerical values

le = LabelEncoder()
le.fit(y_train)

y_train = le.transform(y_train)
y_test = le.transform(y_test)


lr = LogisticRegression()

model = lr.fit(X_train, y_train)
y_train

print("Model coef_=", model.coef_)
print("Model intercept_=", model.intercept_)

train_preds = model.predict(X_train)
train_preds

test_preds = model.predict(X_test)
test_preds


print("train accurracy", accuracy_score(y_train, train_preds))
print("test accurracy", accuracy_score(y_test, test_preds))

cm = confusion_matrix(y_train, train_preds)
print("confusion_matrix for Training = \n", cm)
cm_display = ConfusionMatrixDisplay(cm).plot()

tm = confusion_matrix(y_test, test_preds)
print("confusion_matrix for Testing = \n", tm)
tm_display = ConfusionMatrixDisplay(tm).plot()

print("recall_score for Training = ", recall_score(y_train, train_preds))
print("recall_score for Testing =", recall_score(y_test, test_preds))

print(f1_score(y_train, train_preds))
print(f1_score(y_test, test_preds))

print("precision_score for Training =", precision_score(y_train, train_preds))
print("precision_score for Testing =", precision_score(y_test, test_preds))


def plot(x, y, label):
    plt.scatter(x, y, s=20)
    plt.xlabel("Height", fontsize=15)
    plt.ylabel("Weight", fontsize=15)
  #  cbar= plt.colorbar()
    plt.show()


plt.figure(figsize=(8, 5))
plt.title("Trian data with actual labels", fontsize=15)
plot(df['Height'], df['Weight'], df['Gender'])
plt.show()


# from sklearn.linear_model import LogisticRegression

# data scaling
# create a pipeline of multiple data pre-processing and modeling steps.
# Create a pipeline that includes scaling and linear regression
clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
# Fit the pipeline to the data
clf.fit(X_train, y_train)
# Use decision_function to make binary classification predictions
y_score = clf.decision_function(X_test)
# y1_score = model

fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=clf.classes_[1])
roc_auc = auc(fpr, tpr)

# Calculate the area under the ROC curve (AUC)

plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # plot the random curve
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
