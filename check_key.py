from dotenv import load_dotenv; load_dotenv()
from google import genai
c = genai.Client()
try:
    print('OK:', c.models.generate_content(model='gemini-3.8-flash', contents='hi').text)
except Exception as e:
    print('GOOGLE SAYS:', str(e)[:600])
