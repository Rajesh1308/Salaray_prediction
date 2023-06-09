import streamlit as st
import pickle as pk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Salary_Data.csv")

st.title("Salary Prediction Model")

st.subheader("""This page predicts the salary based on the experience provided. 
The prediction is carried out by Linear regession model trained with a dataset of 31 datas.""")

years = st.number_input(min_value=0, max_value=80, label="Years of Experience")

f = open('model.pkl', 'rb')
model = pk.load(f)
arr = np.array([years]).reshape(-1, 1)
salary = model.predict(arr)
st.write("The predicted salary is Rs." + "%.2f" %salary[0][0])

st.header("Details of Dataset")
st.info("The dataset was taken from Kaggle and contains 31 datas. It was used to train the linear regression model.")
st.line_chart(data)

st.write("The model was built on only one feature which is years of experience. Hence, there will be only one intercept and co-efficient.")
st.latex("y = mx + c")
st.info("Co-effiecient (m) - " + "%.2f" %model.coef_[0][0])            
st.info("Intercept (c) - " + "%.2f" %model.intercept_[0])
x = range(0,12)
m = model.coef_[0][0]
c = model.intercept_[0]

plt.scatter(data['YearsExperience'], data['Salary'])
plt.plot(x, m*x + c, "r-" )
plt.xlabel("Experience (in years)")
plt.ylabel("Salary (in Rupees)")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
st.sidebar.header("Links")
st.sidebar.write("[Source code](https://github.com/Rajesh1308/Salaray_prediction)")
st.sidebar.write("[Dataset](https://www.kaggle.com/datasets/krishnaraj30/salary-prediction-data-simple-linear-regression)")
st.sidebar.header("Dataset")
st.sidebar.dataframe(data, width=250, height=500)
