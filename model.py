import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import streamlit as st


dataset = pd.read_csv("Depression Student Dataset.csv")

# Создаем словари для кодирования
sleep_duration_mapping = {
    'Less than 5 hours': 1,
    '5-6 hours': 2,
    '7-8 hours': 3,
    'More than 8 hours': 4
}

dietary_habits_mapping = {
    'Unhealthy': 1,
    'Moderate': 2,
    'Healthy': 3
}

# Применяем кодирование
dataset['Sleep Duration'] = dataset['Sleep Duration'].map(sleep_duration_mapping)
dataset['Dietary Habits'] = dataset['Dietary Habits'].map(dietary_habits_mapping)

# Преобразуем бинарные переменные с помощью LabelEncoder
label_encoder = LabelEncoder()

dataset['Gender'] = label_encoder.fit_transform(dataset['Gender'])
dataset['Have you ever had suicidal thoughts ?'] = label_encoder.fit_transform(dataset['Have you ever had suicidal thoughts ?'])
dataset['Family History of Mental Illness'] = label_encoder.fit_transform(dataset['Family History of Mental Illness'])
dataset['Depression'] = label_encoder.fit_transform(dataset['Depression'])

# Сохраняем энкодеры в .pkl файл
encoder_dict = {
    'sleep_duration_encoder': sleep_duration_mapping,
    'dietary_habits_encoder': dietary_habits_mapping,
    'gender_encoder': label_encoder,
    'suicidal_thoughts_encoder': label_encoder,
    'family_history_encoder': label_encoder,
    'depression_encoder': label_encoder
}

# Сохранение энкодера в pkl файл
file_path = './encoders.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(encoder_dict, file)


X = dataset.drop(columns='Depression', axis=1)
y = dataset['Depression']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# training using XGBoost
from xgboost import XGBClassifier
xgbclassifier = XGBClassifier()

xgbclassifier.fit(X_train, y_train)
y_pred = xgbclassifier.predict(X_test)
accuracy_score(y_test, y_pred)

# Сохранение лучшей модели в pkl файл
file_path = './model.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(xgbclassifier, file)

