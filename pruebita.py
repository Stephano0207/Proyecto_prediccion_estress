import google.generativeai as genai
import pandas as pd

file_path = "student_lifestyle_dataset.csv"
df = pd.read_csv(file_path)

# Mostrar las primeras filas del dataset
df.head()

