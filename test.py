import os
from dotenv import load_dotenv
from google.generativeai import configure, GenerativeModel, upload_file

# Load API key from .env
load_dotenv()
configure(api_key=os.getenv("GEMINI_API_KEY"))

# Upload the file
file = upload_file("QuestionSampleLucius.mp3")

# Create model (gemini-1.5-* currently supports audio input)
model = GenerativeModel(model_name="models/gemini-2.5-flash")

# Generate content with audio file
response = model.generate_content([
    "Give me the transcription of this audio clip.",
    file
])

# Print the transcription
print(response.text)
