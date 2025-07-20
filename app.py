import streamlit as st
import joblib
import numpy as np

st.title("Employee Salary Prediction App")
st.divider()

st.write("With this app, you can get estimation for the salaries of employees")

Age_Scaled=st.number_input("Enter the age" ,value=1.0,step=0.5,min_value=0.0)
Gender_Encode=st.number_input("Enter the gender",value=0,step=1,min_value=0)
Degree_Encode=st.number_input("Enter the Degree",value=1,step=1,min_value=1)
Job_Title_Encode=st.number_input("Enter the Job-Title",value=1,step=1,min_value=0)
Experience_years_Scaled=st.number_input("Enter the experience" ,value=0.0,step=0.5,min_value=0.0)

X=[Age_Scaled,Gender_Encode,Degree_Encode,Job_Title_Encode, Experience_years_Scaled]
model=joblib.load("linearmodel.pkl")

st.divider()

predict=st.button("Press the button for Salary Prediction")

st.divider()
if predict:
    st.balloons()
    X1=np.array([X])
    prediction=model.predict(X1)[0]
    st.write(f"The predicted salary is: {prediction:,.2f}")

else:
    st.write("Press the button to get the prediction")

