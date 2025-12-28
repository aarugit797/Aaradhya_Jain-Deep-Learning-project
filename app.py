import streamlit as st
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model

#load the model
# load the model safely (no legacy compile info)
model = load_model('model.h5')


with open('ohe_gen.pickle','rb') as file:
    ohe_gen = pickle.load(file)

with open('ohe_geo.pickle','rb') as file:
    ohe_geo = pickle.load(file)

with open('scaler.pickle','rb') as file:
    scaler = pickle.load(file)


#streamlit app
st.title("Customer Churn Prediction")

geography = st.selectbox('Geography', ohe_geo.categories_[0])
gender = st.selectbox('Gender',ohe_gen.categories_[0])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# numerical features
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# one-hot encode geography
geo_encoded = ohe_geo.transform([[geography]])
geo_encoded = pd.DataFrame(
    geo_encoded,
    columns=ohe_geo.get_feature_names_out(['Geography'])
)

# one-hot encode gender
gen_encoded = ohe_gen.transform([[gender]])
gen_encoded = pd.DataFrame(
    gen_encoded,
    columns=ohe_gen.get_feature_names_out(['Gender'])
)

# combine everything
input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded, gen_encoded],
    axis=1
)

input_data = input_data[scaler.feature_names_in_]

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_proba=prediction[0][0]

st.write(f"Probability {prediction_proba}")
if prediction_proba>0.5:
    print("Customer will churn")
else:
    print("Customer will not churn")