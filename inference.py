import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Загрузка модели и энкодеров
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('encoders.pkl', 'rb') as file:
    encoders = pickle.load(file)

# Словари для отображения текста в интерфейсе и числовых значений для модели
gender_options = {0: 'Female', 1: 'Male'}  # Male = 1, Female = 0
sleep_duration_options = {'Less than 5 hours': 1, '5-6 hours': 2, '7-8 hours': 3, 'More than 8 hours': 4}
dietary_habits_options = {'Unhealthy': 1, 'Moderate': 2, 'Healthy': 3}
suicidal_thoughts_options = {0: 'No', 1: 'Yes'}
family_history_options = {0: 'No', 1: 'Yes'}

# Заголовок
st.title('Depression Diagnosis Prediction')

# Ввод данных с помощью Streamlit
age = st.number_input('Age', min_value=18, max_value=100, value=25)

# Ввод через selectbox с отображением текста, но сохранением числовых значений
gender = st.selectbox('Gender', options=['Female', 'Male'])
academic_pressure = st.selectbox('Academic Pressure', options=[1, 2, 3, 4, 5])
study_satisfaction = st.selectbox('Study Satisfaction', options=[1, 2, 3, 4, 5])
sleep_duration = st.selectbox('Sleep Duration', options=['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'])
dietary_habits = st.selectbox('Dietary Habits', options=['Unhealthy', 'Moderate', 'Healthy'])
suicidal_thoughts = st.selectbox('Suicidal Thoughts', options=['No', 'Yes'])
study_hours = st.number_input('Study Hours per Day', min_value=0, max_value=24, value=6)
financial_stress = st.selectbox('Financial Stress', options=[1, 2, 3, 4, 5])
family_history = st.selectbox('Family History of Mental Illness', options=['No', 'Yes'])

# Преобразование ввода в числовые данные для модели
input_data = {
    'Gender': list(gender_options.keys())[list(gender_options.values()).index(gender)],
    'Age': age,
    'Academic Pressure': academic_pressure,  # Порядок и имена точно соответствуют обучающим данным
    'Study Satisfaction': study_satisfaction,
    'Sleep Duration': sleep_duration_options[sleep_duration],
    'Dietary Habits': dietary_habits_options[dietary_habits],
    'Have you ever had suicidal thoughts ?': suicidal_thoughts_options.get(suicidal_thoughts, -1),  # Порядок и имена
    'Study Hours': study_hours,
    'Financial Stress': financial_stress,
    'Family History of Mental Illness': family_history_options.get(family_history, -1)
}

# Создание DataFrame с правильным порядком
input_df = pd.DataFrame([input_data])

# Предсказание с использованием модели
if st.button('Predict'):
    prediction = model.predict(input_df)[0]
    if prediction == 0:
        st.write("Prediction: The person is not depressed.")
    else:
        st.write("Prediction: The person may be depressed.")




#python3 -m streamlit run inference.py

