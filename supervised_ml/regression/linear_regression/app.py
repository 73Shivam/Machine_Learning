import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


st.title("salary prediction")

df = pd.read_csv('Salary_Data.csv')
X = df[['YearsExperience']] # x data that should be taken as a vector or 2D or dataframe
y = df['Salary'] # y data is taken as scaler or 1D or series

xtrain, xtest, ytrain, ytest =  train_test_split(X,y,test_size= .2)

reg = LinearRegression()
reg.fit(xtrain, ytrain) # here the algorithm tries to understand the data

exp = st.number_input('enter your experience in the company')
if st.button('get your salary'):
    data = np.array([exp])

    salary = reg.predict([data])
    st.write("your salary will be "+str(salary[0]))