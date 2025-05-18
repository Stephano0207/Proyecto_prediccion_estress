# test_model.py
import joblib
import pandas as pd

# Cargar modelo y encoder
model = joblib.load('stress_level_model.pkl')
le = joblib.load('label_encoder.pkl')

# Datos de ejemplo (deberían ser similares a tus datos reales)
sample_data = {
    'Study_Hours_Per_Day': 5.2,
    'Extracurricular_Hours_Per_Day': 3.1,
    'Sleep_Hours_Per_Day': 7.5,
    'Social_Hours_Per_Day': 2.3,
    'Physical_Activity_Hours_Per_Day': 1.8,
    'GPA': 16.4
}

# Convertir a DataFrame
df = pd.DataFrame([sample_data.values()], columns=sample_data.keys())

# Predecir
prediction = model.predict(df)
print("Predicted stress level:", le.inverse_transform(prediction)[0])