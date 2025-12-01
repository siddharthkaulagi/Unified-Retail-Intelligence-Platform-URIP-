import google.generativeai as genai

genai.configure(api_key="AIzaSyBerM7vOnykMIMrPeQE_izIH5dfL5d_EGI")

for model in genai.list_models():
    print(model.name)
3