import os, google.generativeai as genai
genai.configure(api_key="AIzaSyAFoP5jNC294oS8ga3_56X8AlK1fPRbsZI")

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
