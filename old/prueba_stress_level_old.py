
import pandas as pd
import joblib

# Cargar el modelo y el codificador
model = joblib.load("stress_level_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Función para predecir el nivel de estrés
def predecir_estres(data_dict):
    columnas = ['Extracurricular_Hours_Per_Day', 'Sleep_Hours_Per_Day',
                'Social_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day', 'GPA']
    datos = pd.DataFrame([data_dict], columns=columnas)
    pred = model.predict(datos)
    nivel_estres = label_encoder.inverse_transform(pred)
    return nivel_estres[0]

# Ejemplo de uso
if __name__ == "__main__":
    ejemplo = {
        'Extracurricular_Hours_Per_Day': 2,
        'Sleep_Hours_Per_Day': 7,
        'Social_Hours_Per_Day': 1,
        'Physical_Activity_Hours_Per_Day': 1,
        'GPA': 14.0
    }
    resultado = predecir_estres(ejemplo)
    print("Nivel de estrés predicho:", resultado)
