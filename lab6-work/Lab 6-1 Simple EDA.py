# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 20:42:57 2023
Exploratory data analysis

@author: cwbong
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load big data into a Pandas dataframe
df = pd.read_csv('./lab6-work/weight-height.csv')

# Check the shape of the dataframe
print(df.shape)

# View the first few rows of the dataframe
print(df.head())

# Check for missing data
print(df.isna().sum())

# Remove missing data
df = df.dropna()

# View descriptive statistics of the dataframe
print(df.describe())

# Create a histogram of a numerical variable
plt.hist(df['Gender'])
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Create a scatter plot of two numerical variables
plt.scatter(df['Weight'], df['Height'])
plt.xlabel('Weight')
plt.ylabel('Height')
plt.show()

# Create a bar chart of a categorical variable
df['Gender'].value_counts().plot(kind='bar')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()
