# app.py
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Configurar Gemini
load_dotenv()  # Carga variables de entorno desde .env
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model_gemini = genai.GenerativeModel('gemini-1.5-flash-latest')  # Más rápido y con mejor cuota





# Cargar el modelo y el label encoder
model = joblib.load('stress_level_model.pkl')
le = joblib.load('label_encoder.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del request
        data = request.get_json()
        
        # Validar que están todos los campos requeridos (6 variables)
        required_fields = [
            'Study_Hours_Per_Day',
            'Extracurricular_Hours_Per_Day',
            'Sleep_Hours_Per_Day',
            'Social_Hours_Per_Day',
            'Physical_Activity_Hours_Per_Day',
            'GPA'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'status': 'error'
                }), 400
        
        # Crear DataFrame con todas las variables originales
        input_data = pd.DataFrame([[
            data['Study_Hours_Per_Day'],
            data['Extracurricular_Hours_Per_Day'],
            data['Sleep_Hours_Per_Day'],
            data['Social_Hours_Per_Day'],
            data['Physical_Activity_Hours_Per_Day'],
            data['GPA']
        ]], columns=[
            'Study_Hours_Per_Day',
            'Extracurricular_Hours_Per_Day',
            'Sleep_Hours_Per_Day',
            'Social_Hours_Per_Day',
            'Physical_Activity_Hours_Per_Day',
            'GPA'
        ])
        
        # Hacer la predicción
        prediction_numeric = model.predict(input_data)
        
        # Convertir el código numérico a la etiqueta original
        # prediction_label = le.inverse_transform(prediction_numeric)
        stress_level = le.inverse_transform(prediction_numeric)[0]
        
        # Generar recomendaciones
        recommendations = generate_recommendations(stress_level, data)


        # Devolver el resultado
        # return jsonify({
        #     'prediction': prediction_label[0],
        #     'status': 'success'
        # })
        return jsonify({
            'prediction': stress_level,
            'recommendations': recommendations,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500



def generate_recommendations(stress_level, features):
    """Genera recomendaciones usando Gemini"""
    prompt = f"""
    Eres un experto psicólogo educativo. Un estudiante tiene un nivel de estrés {stress_level}.
    Basado en estos hábitos:
    - Horas de estudio: {features['Study_Hours_Per_Day']} por día
    - Actividades extracurriculares: {features['Extracurricular_Hours_Per_Day']} horas
    - Horas de sueño: {features['Sleep_Hours_Per_Day']} horas
    - Vida social: {features['Social_Hours_Per_Day']} horas
    - Actividad física: {features['Physical_Activity_Hours_Per_Day']} horas
    - GPA: {features['GPA']}

    Proporciona 3 recomendaciones concretas y personalizadas para mejorar su bienestar.
    Usa un tono empático y profesional. Máximo 200 palabras.
    """
    
    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Recomendaciones no disponibles. Error: {str(e)}"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)