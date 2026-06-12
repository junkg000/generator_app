import os
import google.generativeai as genai

try:
    with open("api_key.txt", "r") as f:
        api_key = f.read().strip()
    
    if not api_key:
        print("API Key is empty.")
    else:
        genai.configure(api_key=api_key)
        print("Available models for this API key:")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
except Exception as e:
    print(f"Error checking models: {e}")
