from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib
# Cargar el dataset
file_path = "student_lifestyle_dataset.csv"
df = pd.read_csv(file_path)

# Mostrar las primeras filas del dataset
df.head()

# 1. Selección de características
features = ['Extracurricular_Hours_Per_Day', 'Sleep_Hours_Per_Day',
            'Social_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day', 'GPA']
X = df[features]

# 2. Codificar la variable objetivo
le = LabelEncoder()
y = le.fit_transform(df['Stress_Level'])  # Low=1, Moderate=2, High=0 (ejemplo)

# 3. División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entrenar el modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Predicción y evaluación
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

print(accuracy, report)
# Guardar el modelo entrenado
joblib.dump(model, 'stress_level_model.pkl')

# Guardar también el codificador de etiquetas (LabelEncoder), por si lo necesitas al predecir
joblib.dump(le, 'label_encoder.pkl')