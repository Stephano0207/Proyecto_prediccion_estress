# app.py
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

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
        prediction_label = le.inverse_transform(prediction_numeric)
        
        # Devolver el resultado
        return jsonify({
            'prediction': prediction_label[0],
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)