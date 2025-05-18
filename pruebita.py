import google.generativeai as genai
import os
from dotenv import load_dotenv
from google.generativeai import list_models

load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

model = genai.GenerativeModel('gemini-1.5-flash-latest')  # Más rápido y con mejor cuota

try:
    response = model.generate_content("Hola en menos de 5 palabras")
    print(response.text)
except Exception as e:
    print(f"Error: {str(e)}")