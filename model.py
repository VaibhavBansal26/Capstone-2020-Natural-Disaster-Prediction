
import numpy as np
import pandas as pd


df_rain = pd.read_csv("Hoppers Crossing-Hourly-Rainfall.csv")
df_rain.head()
df_rain.shape
df_rain.describe()
df_river = pd.read_csv("Hoppers Crossing-Hourly-River-Level.csv")
df_river.head()
df_river.shape
df_river.describe()
df = pd.merge(df_rain, df_river, how='outer', on=['Date/Time'])
df.head()
df['Cumulative rainfall (mm)'] = df['Cumulative rainfall (mm)'].fillna(0)
df['Level (m)'] = df['Level (m)'].fillna(0)

df.head()
df = df.drop(columns=['Current rainfall (mm)', 'Date/Time'])
df.shape
X = df.iloc[:, :1].values
y = df.iloc[:, 1:2].values

x=df.iloc[:, :1]

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression  
lr = LinearRegression()  
lr.fit(X_train, y_train)
print(lr.intercept_)
print(lr.coef_)

# Save your model
from sklearn.externals import joblib
joblib.dump(lr, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
lr = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")